#ifndef GPT4ALL_BINDINGS_GOLANG_BINDING_H_
#define GPT4ALL_BINDINGS_GOLANG_BINDING_H_

#ifndef __cplusplus
#include <stdbool.h>
#else
#include <cstdbool>
#endif

#include "../../gpt4all-backend/llmodel_c.h"

#ifdef __cplusplus
extern "C" {
#endif

llmodel_model load_mpt_model(const char *fname, int n_threads);

llmodel_model load_llama_model(const char *fname, int n_threads);

llmodel_model load_gptj_model(const char *fname, int n_threads);

void gptj_model_prompt(const char *prompt, llmodel_model model, char *result, int repeat_last_n, float repeat_penalty,
                       int n_ctx, int tokens, int top_k, float top_p, float temp, int n_batch, float ctx_erase);

void gptj_free_model(void *state_ptr);

extern bool getTokenCallback(llmodel_model, const char *);

#ifdef __cplusplus
}
#endif

#endif // GPT4ALL_BINDINGS_GOLANG_BINDING_H_
