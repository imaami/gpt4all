#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

#include "llmodel.h"
#include "llmodel_c.h"

struct LLModelWrapper {
    LLModel *llModel = nullptr;
    LLModel::PromptContext promptContext;
};

namespace {
thread_local std::string last_error_message;
} // namespace

llmodel_model llmodel_model_create(const char *model_path) {
    llmodel_error e{};
    auto *model = llmodel_model_create2(model_path, "auto", &e);
    if (!model && e.message)
        std::fprintf(stderr, "%s\n", e.message);
    return model;
}

llmodel_model llmodel_model_create2(const char *model_path, const char *build_variant, llmodel_error *error) {
    auto *wrapper = new LLModelWrapper;
    int errno_value = 0;

    try {
        wrapper->llModel = LLModel::construct(model_path, build_variant);
    } catch (const std::exception& e) {
        errno_value = EINVAL;
        last_error_message = e.what();
    }

    if (!wrapper->llModel) {
        delete std::exchange(wrapper, nullptr);
        // Get errno code and message if they're still unset
        if (!errno_value && (errno_value = errno))
            last_error_message = std::strerror(errno_value);
    }

    // On failure report back the reason if possible
    if (errno_value && error) {
        error->message = last_error_message.c_str();
        error->code = errno_value;
    }

    return reinterpret_cast<llmodel_model>(wrapper);
}

void llmodel_model_destroy(llmodel_model *model) {
    if (model) {
        auto *wrapper = reinterpret_cast<LLModelWrapper *>(*model);
        if (wrapper)
            delete wrapper->llModel;
        *model = nullptr;
    }
}

int llmodel_model_load(llmodel_model model, const char *model_path)
{
    auto *wrapper = reinterpret_cast<LLModelWrapper *>(model);
    return static_cast<int>(wrapper->llModel->loadModel(model_path));
}

int llmodel_model_is_loaded(llmodel_model model)
{
    auto *wrapper = reinterpret_cast<LLModelWrapper *>(model);
    return static_cast<int>(wrapper->llModel->isModelLoaded());
}

std::size_t llmodel_state_get_size(llmodel_model model)
{
    return reinterpret_cast<LLModelWrapper *>(model)->llModel->stateSize();
}

std::size_t llmodel_state_save(llmodel_model model, unsigned char *dest, std::size_t)
{
    return reinterpret_cast<LLModelWrapper *>(model)->llModel->saveState(dest);
}

std::size_t llmodel_state_restore(llmodel_model model, unsigned char const *src, std::size_t)
{
    return reinterpret_cast<LLModelWrapper *>(model)->llModel->restoreState(src);
}

void llmodel_prompt(llmodel_model model, const char *prompt,
                    llmodel_prompt_cb *prompt_cb,
                    llmodel_response_cb *response_cb,
                    llmodel_recalculate_cb *recalculate_cb,
                    llmodel_prompt_context *ctx)
{
    auto *wrapper = reinterpret_cast<LLModelWrapper *>(model);

    // Copy the C prompt context
    wrapper->promptContext.n_past = ctx->n_past;
    wrapper->promptContext.n_ctx = ctx->n_ctx;
    wrapper->promptContext.n_predict = ctx->n_predict;
    wrapper->promptContext.top_k = ctx->top_k;
    wrapper->promptContext.top_p = ctx->top_p;
    wrapper->promptContext.temp = ctx->temp;
    wrapper->promptContext.n_batch = ctx->n_batch;
    wrapper->promptContext.repeat_penalty = ctx->repeat_penalty;
    wrapper->promptContext.repeat_last_n = ctx->repeat_last_n;
    wrapper->promptContext.contextErase = ctx->context_erase;

    // Call the C++ prompt method
    wrapper->llModel->prompt(
        prompt,
        [cb = prompt_cb](std::int32_t tok_id) {
            return 0 != cb(tok_id);
        },
        [cb = response_cb](std::int32_t tok_id, const std::string &resp) {
            return 0 != cb(tok_id, resp.c_str());
        },
        [cb = recalculate_cb](bool is_recalc) {
            return 0 != cb(static_cast<int>(is_recalc));
        },
        wrapper->promptContext
    );

    // Update the C context by giving access to the wrappers raw pointers to std::vector data
    // which involves no copies
    ctx->logits = wrapper->promptContext.logits.data();
    ctx->logits_size = wrapper->promptContext.logits.size();
    ctx->tokens = wrapper->promptContext.tokens.data();
    ctx->tokens_size = wrapper->promptContext.tokens.size();

    // Update the rest of the C prompt context
    ctx->n_past = wrapper->promptContext.n_past;
    ctx->n_ctx = wrapper->promptContext.n_ctx;
    ctx->n_predict = wrapper->promptContext.n_predict;
    ctx->top_k = wrapper->promptContext.top_k;
    ctx->top_p = wrapper->promptContext.top_p;
    ctx->temp = wrapper->promptContext.temp;
    ctx->n_batch = wrapper->promptContext.n_batch;
    ctx->repeat_penalty = wrapper->promptContext.repeat_penalty;
    ctx->repeat_last_n = wrapper->promptContext.repeat_last_n;
    ctx->context_erase = wrapper->promptContext.contextErase;
}

void llmodel_set_thread_count(llmodel_model model, std::int32_t n_threads)
{
    reinterpret_cast<LLModelWrapper *>(model)->llModel->setThreadCount(n_threads);
}

std::int32_t llmodel_get_thread_count(llmodel_model model)
{
    return reinterpret_cast<LLModelWrapper *>(model)->llModel->threadCount();
}
