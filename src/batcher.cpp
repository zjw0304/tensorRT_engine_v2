#include <trt_engine/batcher.h>

#include <algorithm>
#include <chrono>

namespace trt_engine {

// ── Constructor / Destructor ────────────────────────────────────────────

DynamicBatcher::DynamicBatcher(std::shared_ptr<InferenceEngine> engine,
                               int max_batch_size,
                               int max_wait_time_ms)
    : engine_(std::move(engine)),
      max_batch_size_(max_batch_size),
      max_wait_time_ms_(max_wait_time_ms) {

    if (!engine_) {
        throw EngineException("DynamicBatcher: engine pointer is null");
    }
    if (max_batch_size_ < 1) {
        throw EngineException("DynamicBatcher: max_batch_size must be >= 1");
    }
    if (max_wait_time_ms_ < 0) {
        throw EngineException("DynamicBatcher: max_wait_time_ms must be >= 0");
    }

    batch_thread_ = std::thread(&DynamicBatcher::batch_loop, this);
    get_logger().info("DynamicBatcher started: max_batch=" +
                      std::to_string(max_batch_size_) +
                      " max_wait_ms=" + std::to_string(max_wait_time_ms_));
}

DynamicBatcher::~DynamicBatcher() {
    shutdown_.store(true);
    queue_cv_.notify_all();
    if (batch_thread_.joinable()) {
        batch_thread_.join();
    }
}

// ── Submit ──────────────────────────────────────────────────────────────

std::future<InferenceResult> DynamicBatcher::submit(
    const std::vector<std::vector<float>>& single_input) {

    auto req = std::make_shared<PendingRequest>();
    req->inputs = single_input;
    auto future = req->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (shutdown_.load()) {
            req->promise.set_value(InferenceResult{
                {}, 0.0f, false, "DynamicBatcher is shut down"});
            return future;
        }
        pending_queue_.push(std::move(req));
    }
    queue_cv_.notify_one();
    return future;
}

// ── Batch loop ──────────────────────────────────────────────────────────

void DynamicBatcher::batch_loop() {
    while (true) {
        std::vector<std::shared_ptr<PendingRequest>> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait until at least one request arrives or shutdown
            queue_cv_.wait(lock, [this]{
                return shutdown_.load() || !pending_queue_.empty();
            });

            if (shutdown_.load() && pending_queue_.empty()) return;

            // Start collecting a batch
            auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(max_wait_time_ms_);

            while (static_cast<int>(batch.size()) < max_batch_size_) {
                // Take whatever is available right now
                while (!pending_queue_.empty() &&
                       static_cast<int>(batch.size()) < max_batch_size_) {
                    batch.push_back(std::move(pending_queue_.front()));
                    pending_queue_.pop();
                }

                if (static_cast<int>(batch.size()) >= max_batch_size_) break;

                // Wait for more or until deadline
                auto status = queue_cv_.wait_until(lock, deadline);
                if (status == std::cv_status::timeout) break;
                if (shutdown_.load()) break;
            }
        }

        if (batch.empty()) continue;

        execute_batch(batch);
    }
}

// ── Execute batch ───────────────────────────────────────────────────────

void DynamicBatcher::execute_batch(
    std::vector<std::shared_ptr<PendingRequest>>& batch) {

    int batch_size = static_cast<int>(batch.size());

    if (batch_size == 0) return;

    // For a single request, just run it directly
    if (batch_size == 1) {
        auto result = engine_->infer(batch[0]->inputs);
        batch[0]->promise.set_value(std::move(result));
        return;
    }

    // Concatenate individual sample inputs into batch tensors.
    // Assumes each request has the same number of input tensors and
    // each tensor for a single sample has the same element count.
    size_t num_tensors = batch[0]->inputs.size();

    // Verify consistency
    for (auto& req : batch) {
        if (req->inputs.size() != num_tensors) {
            // All requests in this batch get an error -- cannot form a valid batch
            for (auto& r : batch) {
                InferenceResult err_result;
                err_result.success = false;
                err_result.error_msg = "Inconsistent number of input tensors in batch";
                r->promise.set_value(std::move(err_result));
            }
            return;
        }
    }

    // Determine per-sample element counts per tensor
    std::vector<size_t> per_sample_sizes(num_tensors);
    for (size_t t = 0; t < num_tensors; ++t) {
        per_sample_sizes[t] = batch[0]->inputs[t].size();
    }

    // Build the batched input by concatenating along the leading (batch) dim
    std::vector<std::vector<float>> batched_inputs(num_tensors);
    for (size_t t = 0; t < num_tensors; ++t) {
        batched_inputs[t].reserve(per_sample_sizes[t] * batch_size);
        for (auto& req : batch) {
            batched_inputs[t].insert(batched_inputs[t].end(),
                                     req->inputs[t].begin(),
                                     req->inputs[t].end());
        }
    }

    // Set the batch dimension shape on the engine before inference.
    // Infer the correct dims: take the engine input info, replace dim 0 with batch_size.
    auto input_infos = engine_->get_input_info();
    for (size_t t = 0; t < num_tensors && t < input_infos.size(); ++t) {
        auto dims = input_infos[t].shape;
        if (!dims.empty()) {
            dims[0] = batch_size;
            engine_->set_input_shape(input_infos[t].name, dims);
        }
    }

    // Run batched inference
    auto batch_result = engine_->infer(batched_inputs);

    if (!batch_result.success) {
        // Forward error to all requests
        for (auto& req : batch) {
            InferenceResult r;
            r.success = false;
            r.error_msg = batch_result.error_msg;
            req->promise.set_value(std::move(r));
        }
        return;
    }

    // Split batch results back to individual futures.
    // Each output tensor is sliced by per_sample_output_size.
    size_t num_outputs = batch_result.outputs.size();

    for (int b = 0; b < batch_size; ++b) {
        InferenceResult individual;
        individual.success = true;
        individual.latency_ms = batch_result.latency_ms;
        individual.outputs.resize(num_outputs);

        for (size_t o = 0; o < num_outputs; ++o) {
            size_t total = batch_result.outputs[o].size();
            size_t per_sample = total / static_cast<size_t>(batch_size);
            size_t start = static_cast<size_t>(b) * per_sample;
            size_t end = start + per_sample;
            if (end > total) end = total;
            individual.outputs[o].assign(
                batch_result.outputs[o].begin() + static_cast<std::ptrdiff_t>(start),
                batch_result.outputs[o].begin() + static_cast<std::ptrdiff_t>(end));
        }

        batch[b]->promise.set_value(std::move(individual));
    }
}

}  // namespace trt_engine
