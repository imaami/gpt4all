#include "binding.h"

#include "../../gpt4all-backend/llmodel.h"
#include "../../gpt4all-backend/llama.cpp/llama.h"
#include "../../gpt4all-backend/llmodel_c.cpp"
#include "../../gpt4all-backend/mpt.h"
#include "../../gpt4all-backend/mpt.cpp"

#include "../../gpt4all-backend/llamamodel.h"
#include "../../gpt4all-backend/gptj.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>

llmodel_model load_mpt_model(const char *fname, int n_threads) {
    // load the model
    auto gptj = llmodel_mpt_create();

    llmodel_setThreadCount(gptj,  n_threads);
    if (!llmodel_loadModel(gptj, fname)) {
        return nullptr;
    }

    return gptj;
}

llmodel_model load_llama_model(const char *fname, int n_threads) {
    // load the model
    auto gptj = llmodel_llama_create();

    llmodel_setThreadCount(gptj,  n_threads);
    if (!llmodel_loadModel(gptj, fname)) {
        return nullptr;
    }

    return gptj;
}

llmodel_model load_gptj_model(const char *fname, int n_threads) {
    // load the model
    auto gptj = llmodel_gptj_create();

    llmodel_setThreadCount(gptj,  n_threads);
    if (!llmodel_loadModel(gptj, fname)) {
        return nullptr;
    }

    return gptj;
}

void gptj_model_prompt(const char *prompt, llmodel_model model, char *result, int repeat_last_n, float repeat_penalty,
                       int n_ctx, int tokens, int top_k, float top_p, float temp, int n_batch, float ctx_erase)
{
    std::string res{};

    auto lambda_response = [model, &res](std::int32_t, const char *responsechars) -> bool { res.append(responsechars); return !!getTokenCallback(model, responsechars); };
	auto lambda_recalculate = [](bool is_recalculating) {
	        // You can handle recalculation requests here if needed
	    return is_recalculating;
	};

    llmodel_prompt_context prompt_ctx{};

    prompt_ctx.n_ctx          = 1024;
    prompt_ctx.n_predict      = 50;
    prompt_ctx.top_k          = 10;
    prompt_ctx.top_p          = 0.9f;
    prompt_ctx.temp           = 1.0f;
    prompt_ctx.n_batch        = 1;
    prompt_ctx.repeat_penalty = 1.2f;
    prompt_ctx.repeat_last_n  = 10;
    prompt_ctx.context_erase  = 0.5f;

    prompt_ctx.n_predict = tokens;
    prompt_ctx.repeat_last_n = repeat_last_n;
    prompt_ctx.repeat_penalty = repeat_penalty;
    prompt_ctx.n_ctx = n_ctx;
    prompt_ctx.top_k = top_k;
    prompt_ctx.context_erase = ctx_erase;
    prompt_ctx.top_p = top_p;
    prompt_ctx.temp = temp;
    prompt_ctx.n_batch = n_batch;

    llmodel_prompt(model, prompt,
                   [](std::int32_t) -> bool { return true; },
                   lambda_response,
                   lambda_recalculate,
                   &prompt_ctx);

    strcpy(result, res.c_str()); 
}

void gptj_free_model(void *state_ptr) {
    llmodel_model* ctx = (llmodel_model*) state_ptr;
    llmodel_llama_destroy(ctx);
}

