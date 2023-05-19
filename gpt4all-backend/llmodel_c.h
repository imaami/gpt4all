#ifndef LLMODEL_C_H
#define LLMODEL_C_H

#ifndef __cplusplus
# include <stddef.h>
# include <stdint.h>
# define STD_(x) x
#else
# include <cstddef>
# include <cstdint>
# define STD_(x) ::std::x
extern "C" {
#endif

/**
 * Opaque pointer to the underlying model.
 */
typedef void *llmodel_model;

/**
 * Structure containing any errors that may eventually occur
 */
struct llmodel_error {
    const char *message;  //!< Human readable error description; Thread-local; guaranteed to survive until next llmodel C API call
    int code;             //!< errno; 0 if none
};
#ifndef __cplusplus
typedef struct llmodel_error llmodel_error;
#endif

/**
 * llmodel_prompt_context structure for holding the prompt context.
 * NOTE: The implementation takes care of all the memory handling of the raw logits pointer and the
 * raw tokens pointer. Attempting to resize them or modify them in any way can lead to undefined
 * behavior.
 */
struct llmodel_prompt_context {
    float *logits;               //!< logits of current context
    STD_(size_t) logits_size;    //!< the size of the raw logits vector
    STD_(int32_t) *tokens;       //!< current tokens in the context window
    STD_(size_t) tokens_size;    //!< the size of the raw tokens vector
    STD_(int32_t) n_past;        //!< number of tokens in past conversation
    STD_(int32_t) n_ctx;         //!< number of tokens possible in context window
    STD_(int32_t) n_predict;     //!< number of tokens to predict
    STD_(int32_t) top_k;         //!< top k logits to sample from
    float top_p;                 //!< nucleus sampling probability threshold
    float temp;                  //!< temperature to adjust model's output distribution
    STD_(int32_t) n_batch;       //!< number of predictions to generate in parallel
    float repeat_penalty;        //!< penalty factor for repeated tokens
    STD_(int32_t) repeat_last_n; //!< last n tokens to penalize
    float context_erase;         //!< percent of context to erase if we exceed the context window
};
#ifndef __cplusplus
typedef struct llmodel_prompt_context llmodel_prompt_context;
#endif

/**
 * Callback type for prompt processing.
 * @param token_id The token id of the prompt.
 *
 * @return A boolean integer value indicating whether the model should keep processing.
 * @retval 1 processing should continue.
 * @retval 0 processing should stop.
 */
typedef int (llmodel_prompt_cb)(STD_(int32_t) token_id);

/**
 * Callback type for response.
 * @param token_id The token id of the response.
 * @param response The response string. NOTE: a token_id of -1 indicates the string is an error string.
 *
 * @return A boolean integer value indicating whether the model should keep generating.
 * @retval 1 generating should continue.
 * @retval 0 generating should stop.
 */
typedef int (llmodel_response_cb)(STD_(int32_t) token_id, const char *response);

/**
 * Callback type for recalculation of context.
 * @param is_recalculating whether the model is recalculating the context.
 *
 * @return A boolean integer value indicating whether the model should keep generating.
 * @retval 1 generating should continue.
 * @retval 0 generating should stop.
 */
typedef int (llmodel_recalculate_cb)(int is_recalculating);

/**
 * Create an llmodel instance.
 * Recognises correct model type from file at model_path
 * @param model_path A string representing the path to the model file.
 * @return An llmodel_model instance handle (i.e. a void pointer).
 * @retval NULL if model creation failed.
 */
extern llmodel_model llmodel_model_create(const char *model_path) __attribute__ ((deprecated));

/**
 * Create an llmodel instance.
 * Recognises correct model type from file at model_path
 * @param model_path A string representing the path to the model file; will only be used to detect model type.
 * @param build_variant A string representing the implementation to use (auto, default, avxonly, ...),
 * @param error A pointer to a llmodel_error; will only be set on error.
 * @return An llmodel_model instance handle (i.e. a void pointer).
 * @retval NULL if model creation failed.
 */
extern llmodel_model llmodel_model_create2(const char *model_path, const char *build_variant, llmodel_error *error);

/**
 * Destroy an llmodel instance.
 * Recognises correct model type using type info
 * @param model A pointer to an opaque @ref llmodel_model instance handle.
 */
extern void llmodel_model_destroy(llmodel_model *model);

/**
 * Load a model from a file.
 * @param model An opaque @ref llmodel_model instance handle.
 * @param model_path The path to the model file to load.
 * @return 1 if the model was loaded successfully, 0 otherwise.
 */
extern int llmodel_model_load(llmodel_model model, const char *model_path);

/**
 * Check if a model is loaded.
 * @param model An opaque @ref llmodel_model instance handle.
 * @return 1 if the model is loaded, 0 otherwise.
 */
extern int llmodel_model_is_loaded(llmodel_model model);

/**
 * Get the size of the internal state of the model.
 * NOTE: This state data is specific to the type of model you have created.
 * @param model An opaque @ref llmodel_model instance handle.
 * @return the size in bytes of the internal state of the model
 */
extern STD_(size_t) llmodel_state_get_size(llmodel_model model);

/**
 * Saves the internal state of the model to the specified destination address.
 * NOTE: This state data is specific to the type of model you have created.
 * @param model An opaque @ref llmodel_model instance handle.
 * @param dest A pointer to the destination buffer.
 * @param dest_len The size of the destination buffer.
 * @return the number of bytes copied
 */
extern STD_(size_t) llmodel_state_save(llmodel_model model, unsigned char *dest, STD_(size_t) dest_len);

/**
 * Restores the internal state of the model using data from the specified address.
 * NOTE: This state data is specific to the type of model you have created.
 * @param model An opaque @ref llmodel_model instance handle.
 * @param src A pointer to the state data to restore.
 * @param src_len The size of the source data buffer.
 * @return the number of bytes read
 */
extern STD_(size_t) llmodel_state_restore(llmodel_model model, unsigned char const *src, STD_(size_t) src_len);

/**
 * Generate a response using the model.
 * @param model An opaque @ref llmodel_model instance handle.
 * @param prompt The input prompt string.
 * @param prompt_cb A callback function for handling the processing of prompt.
 * @param response_cb A callback function for handling the generated response.
 * @param recalculate_cb A callback function for handling recalculation requests.
 * @param ctx A pointer to the llmodel_prompt_context structure.
 */
extern void llmodel_prompt(llmodel_model model, const char *prompt,
                           llmodel_prompt_cb *prompt_cb,
                           llmodel_response_cb *response_cb,
                           llmodel_recalculate_cb *recalculate_cb,
                           llmodel_prompt_context *ctx);

/**
 * Set the number of threads to be used by the model.
 * @param model An opaque @ref llmodel_model instance handle.
 * @param n_threads The number of threads to be used.
 */
extern void llmodel_set_thread_count(llmodel_model model, STD_(int32_t) n_threads);

/**
 * Get the number of threads currently being used by the model.
 * @param model An opaque @ref llmodel_model instance handle.
 * @return The number of threads currently being used.
 */
extern STD_(int32_t) llmodel_get_thread_count(llmodel_model model);

#ifdef __cplusplus
}
#endif

#undef STD_

#endif // LLMODEL_C_H
