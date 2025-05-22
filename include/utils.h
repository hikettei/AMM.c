#include <assert.h>
#define amm_assert(condition, message, ...)     \
  do {                                          \
    if (!(condition)) {                         \
      fprintf(stderr, message, ##__VA_ARGS__);  \
      fprintf(stderr, " (");                    \
      assert(condition);                        \
      fprintf(stderr, ")\n");                   \
    }                                           \
  } while (0)

#define AMM_UTILS_H
#if defined(__GNUC__) && !defined(__clang__)
#define AMM_C_GCC_MODE
#elif defined(__clang__) && !defined(__cplusplus__)
#define AMM_C_BLOCK_MODE
#else
#error "AMM.c: Cannot compile with this compiler. Please use GCC or Clang."
#endif

#if defined(AMM_C_GCC_MODE)
#  define amm_lambda(ret_type, ...)             \
  __extension__                                 \
  ({                                            \
    ret_type __fn__ __VA_ARGS__                 \
      __fn__                                    \
  })
typedef void (*amm_noarg_callback)(void);
#elif defined(AMM_C_BLOCK_MODE)
#  if !__has_feature(blocks)
#    error "AMM.c: To build AMM.c with Apple Clang, you need to use the -fblocks flag."
#  endif
#  define amm_lambda(ret_type, ...)             \
  ^ret_type __VA_ARGS__
typedef void (^amm_noarg_callback)(void);
#else
#  error "AMM.c: Cannot compile with this compiler. Please use GCC or Clang."
#endif
