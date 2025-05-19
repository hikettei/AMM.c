#include <assert.h>
#define amm_assert(condition, message) assert(condition && message)

#if defined(__GNUC__) && !defined(__clang__)
#  define amm_lambda(ret_type, ...)             \
  __extension__                                 \
  ({                                            \
    ret_type __fn__ __VA_ARGS__                 \
      __fn__;                                   \
  })
typedef void (*amm_noarg_callback)(void);
#elif defined(__clang__) && !defined(__cplusplus__)
#  if !__has_feature(blocks)
#    error "AMM.c: To build AMM.c with Apple Clang, you need to use the -fblocks flag."
#  endif
#  define amm_lambda(ret_type, ...)             \
  ^ret_type __VA_ARGS__
typedef void (^amm_noarg_callback)(void);
#else
#  error "AMM.c: Cannot compile with this compiler. Please use GCC or Clang."
#endif
