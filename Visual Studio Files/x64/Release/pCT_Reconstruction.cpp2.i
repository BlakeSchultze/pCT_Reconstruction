#line 1 "x64/Release/pCT_Reconstruction.cudafe1.gpu"
typedef char __nv_bool;
struct __C1;struct __C2;struct __C3;struct __C4;struct __C5;union __C6;struct __C7;struct __type_info;struct __fundamental_type_info;struct __class_type_info;struct __si_class_type_info;
#line 56 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
enum _ZN13vc_attributes10YesNoMaybeE {
#line 59 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes2NoE = 268369921,
#line 60 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes5MaybeE = 268369936,
#line 61 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes3YesE = 268370176};
#line 66 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
enum _ZN13vc_attributes10AccessTypeE {
#line 68 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes8NoAccessE,
#line 69 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes4ReadE,
#line 70 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes5WriteE,
#line 71 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
_ZN13vc_attributes9ReadWriteE};
#line 1388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"
struct CUstream_st;
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
struct _iobuf;
#line 211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUipcMem_flags_enum {
#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1};
#line 220 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUmemAttach_flags_enum {
#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEM_ATTACH_GLOBAL = 1,
#line 222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEM_ATTACH_HOST,
#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEM_ATTACH_SINGLE = 4};
#line 229 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUctx_flags_enum {
#line 230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_SCHED_AUTO,
#line 231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_SCHED_SPIN,
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_SCHED_YIELD,
#line 233 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_SCHED_BLOCKING_SYNC = 4,
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_BLOCKING_SYNC = 4,
#line 237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_SCHED_MASK = 7,
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_MAP_HOST,
#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_LMEM_RESIZE_TO_MAX = 16,
#line 240 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CTX_FLAGS_MASK = 31};
#line 246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUstream_flags_enum {
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_STREAM_DEFAULT,
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_STREAM_NON_BLOCKING};
#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUevent_flags_enum {
#line 255 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_EVENT_DEFAULT,
#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_EVENT_BLOCKING_SYNC,
#line 257 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_EVENT_DISABLE_TIMING,
#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_EVENT_INTERPROCESS = 4};
#line 264 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUarray_format_enum {
#line 265 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_UNSIGNED_INT8 = 1,
#line 266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_UNSIGNED_INT16,
#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_UNSIGNED_INT32,
#line 268 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_SIGNED_INT8 = 8,
#line 269 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_SIGNED_INT16,
#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_SIGNED_INT32,
#line 271 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_HALF = 16,
#line 272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_AD_FORMAT_FLOAT = 32};
#line 278 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUaddress_mode_enum {
#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TR_ADDRESS_MODE_WRAP,
#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TR_ADDRESS_MODE_CLAMP,
#line 281 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TR_ADDRESS_MODE_MIRROR,
#line 282 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TR_ADDRESS_MODE_BORDER};
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUfilter_mode_enum {
#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TR_FILTER_MODE_POINT,
#line 290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TR_FILTER_MODE_LINEAR};
#line 296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUdevice_attribute_enum {
#line 297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
#line 299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
#line 300 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
#line 301 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
#line 302 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
#line 304 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
#line 305 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_WARP_SIZE,
#line 308 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_PITCH,
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
#line 310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
#line 313 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
#line 314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
#line 316 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_INTEGRATED,
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
#line 318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
#line 319 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
#line 320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
#line 322 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
#line 323 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
#line 325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
#line 326 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
#line 328 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
#line 329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,
#line 331 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
#line 332 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
#line 335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
#line 336 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
#line 337 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
#line 338 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
#line 340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
#line 341 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
#line 342 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
#line 343 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
#line 344 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
#line 347 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
#line 349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
#line 351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
#line 352 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
#line 353 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
#line 354 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
#line 355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
#line 357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
#line 358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
#line 359 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
#line 360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
#line 361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
#line 362 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
#line 365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
#line 366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
#line 371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
#line 373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
#line 374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
#line 377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
#line 379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX};
#line 409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUpointer_attribute_enum {
#line 410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_CONTEXT = 1,
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
#line 412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
#line 413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_HOST_POINTER,
#line 414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_P2P_TOKENS,
#line 415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_BUFFER_ID,
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_POINTER_ATTRIBUTE_IS_MANAGED};
#line 423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUfunction_attribute_enum {
#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
#line 436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
#line 447 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
#line 452 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_NUM_REGS,
#line 461 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_PTX_VERSION,
#line 470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_BINARY_VERSION,
#line 476 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
#line 478 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_ATTRIBUTE_MAX};
#line 484 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUfunc_cache_enum {
#line 485 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_CACHE_PREFER_NONE,
#line 486 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_CACHE_PREFER_SHARED,
#line 487 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_CACHE_PREFER_L1,
#line 488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_FUNC_CACHE_PREFER_EQUAL};
#line 494 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUsharedconfig_enum {
#line 495 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
#line 496 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE};
#line 503 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUmemorytype_enum {
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEMORYTYPE_HOST = 1,
#line 505 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEMORYTYPE_DEVICE,
#line 506 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEMORYTYPE_ARRAY,
#line 507 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_MEMORYTYPE_UNIFIED};
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUcomputemode_enum {
#line 514 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_COMPUTEMODE_DEFAULT,
#line 515 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_COMPUTEMODE_EXCLUSIVE,
#line 516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_COMPUTEMODE_PROHIBITED,
#line 517 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_COMPUTEMODE_EXCLUSIVE_PROCESS};
#line 523 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUjit_option_enum {
#line 530 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_MAX_REGISTERS,
#line 545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_THREADS_PER_BLOCK,
#line 553 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_WALL_TIME,
#line 562 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INFO_LOG_BUFFER,
#line 571 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
#line 580 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_ERROR_LOG_BUFFER,
#line 589 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
#line 597 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_OPTIMIZATION_LEVEL,
#line 605 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_TARGET_FROM_CUCONTEXT,
#line 613 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_TARGET,
#line 621 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_FALLBACK_STRATEGY,
#line 629 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_GENERATE_DEBUG_INFO,
#line 636 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_LOG_VERBOSE,
#line 643 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_GENERATE_LINE_INFO,
#line 651 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_CACHE_MODE,
#line 653 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_NUM_OPTIONS};
#line 660 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUjit_target_enum {
#line 662 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_10 = 10,
#line 663 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_11,
#line 664 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_12,
#line 665 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_13,
#line 666 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_20 = 20,
#line 667 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_21,
#line 668 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_30 = 30,
#line 669 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_32 = 32,
#line 670 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_35 = 35,
#line 671 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_TARGET_COMPUTE_50 = 50};
#line 677 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUjit_fallback_enum {
#line 679 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_PREFER_PTX,
#line 681 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_PREFER_BINARY};
#line 688 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUjit_cacheMode_enum {
#line 690 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_CACHE_OPTION_NONE,
#line 691 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_CACHE_OPTION_CG,
#line 692 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_CACHE_OPTION_CA};
#line 698 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUjitInputType_enum {
#line 704 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INPUT_CUBIN,
#line 710 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INPUT_PTX,
#line 716 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INPUT_FATBINARY,
#line 722 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INPUT_OBJECT,
#line 728 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_INPUT_LIBRARY,
#line 730 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_JIT_NUM_INPUT_TYPES};
#line 740 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUgraphicsRegisterFlags_enum {
#line 741 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_NONE,
#line 742 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
#line 743 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
#line 744 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
#line 745 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8};
#line 751 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUgraphicsMapResourceFlags_enum {
#line 752 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
#line 753 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY,
#line 754 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD};
#line 760 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUarray_cubemap_face_enum {
#line 761 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CUBEMAP_FACE_POSITIVE_X,
#line 762 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_X,
#line 763 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Y,
#line 764 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Y,
#line 765 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Z,
#line 766 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Z};
#line 772 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUlimit_enum {
#line 773 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_LIMIT_STACK_SIZE,
#line 774 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_LIMIT_PRINTF_FIFO_SIZE,
#line 775 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_LIMIT_MALLOC_HEAP_SIZE,
#line 776 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
#line 777 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
#line 778 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_LIMIT_MAX};
#line 784 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUresourcetype_enum {
#line 785 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RESOURCE_TYPE_ARRAY,
#line 786 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
#line 787 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RESOURCE_TYPE_LINEAR,
#line 788 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RESOURCE_TYPE_PITCH2D};
#line 794 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum cudaError_enum {
#line 800 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_SUCCESS,
#line 806 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_VALUE,
#line 812 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_OUT_OF_MEMORY,
#line 818 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_INITIALIZED,
#line 823 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_DEINITIALIZED,
#line 830 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PROFILER_DISABLED,
#line 838 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PROFILER_NOT_INITIALIZED,
#line 845 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PROFILER_ALREADY_STARTED,
#line 852 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PROFILER_ALREADY_STOPPED,
#line 858 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NO_DEVICE = 100,
#line 864 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_DEVICE,
#line 871 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_IMAGE = 200,
#line 881 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_CONTEXT,
#line 890 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
#line 895 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_MAP_FAILED = 205,
#line 900 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_UNMAP_FAILED,
#line 906 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ARRAY_IS_MAPPED,
#line 911 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ALREADY_MAPPED,
#line 919 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NO_BINARY_FOR_GPU,
#line 924 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ALREADY_ACQUIRED,
#line 929 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_MAPPED,
#line 935 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
#line 941 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_POINTER,
#line 947 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ECC_UNCORRECTABLE,
#line 953 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_UNSUPPORTED_LIMIT,
#line 960 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
#line 966 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
#line 971 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_PTX,
#line 976 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_SOURCE = 300,
#line 981 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_FILE_NOT_FOUND,
#line 986 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
#line 991 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
#line 996 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_OPERATING_SYSTEM,
#line 1002 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_HANDLE = 400,
#line 1008 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_FOUND = 500,
#line 1016 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_READY = 600,
#line 1025 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ILLEGAL_ADDRESS = 700,
#line 1036 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
#line 1047 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_LAUNCH_TIMEOUT,
#line 1053 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
#line 1060 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
#line 1067 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
#line 1073 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
#line 1080 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_CONTEXT_IS_DESTROYED,
#line 1088 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ASSERT,
#line 1095 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_TOO_MANY_PEERS,
#line 1101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
#line 1107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
#line 1116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_HARDWARE_STACK_ERROR,
#line 1124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_ILLEGAL_INSTRUCTION,
#line 1133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_MISALIGNED_ADDRESS,
#line 1144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_ADDRESS_SPACE,
#line 1152 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_INVALID_PC,
#line 1162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_LAUNCH_FAILED,
#line 1168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_PERMITTED = 800,
#line 1174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_NOT_SUPPORTED,
#line 1179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CUDA_ERROR_UNKNOWN = 999};
#line 1408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
enum CUresourceViewFormat_enum {
#line 1410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_NONE,
#line 1411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_1X8,
#line 1412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_2X8,
#line 1413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_4X8,
#line 1414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_1X8,
#line 1415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_2X8,
#line 1416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_4X8,
#line 1417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_1X16,
#line 1418 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_2X16,
#line 1419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_4X16,
#line 1420 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_1X16,
#line 1421 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_2X16,
#line 1422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_4X16,
#line 1423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_1X32,
#line 1424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_2X32,
#line 1425 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_4X32,
#line 1426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_1X32,
#line 1427 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_2X32,
#line 1428 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_4X32,
#line 1429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_1X16,
#line 1430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_2X16,
#line 1431 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_4X16,
#line 1432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_1X32,
#line 1433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_2X32,
#line 1434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_4X32,
#line 1435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC1,
#line 1436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC2,
#line 1437 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC3,
#line 1438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC4,
#line 1439 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SIGNED_BC4,
#line 1440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC5,
#line 1441 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SIGNED_BC5,
#line 1442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H,
#line 1443 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_SIGNED_BC6H,
#line 1444 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC7};
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\typeinfo"
struct type_info;
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
struct _Ctypevec;
#line 65 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
struct _Cvtvec;
#line 112 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum DATA_FORMATS {
#line 112 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
OLD_FORMAT,
#line 112 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
VERSION_0,
#line 112 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
VERSION_1};
#line 135 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum BIN_ANALYSIS_TYPE {
#line 135 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
MEANS,
#line 135 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
COUNTS,
#line 135 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
MEMBERS};
#line 136 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum BIN_ANALYSIS_FOR {
#line 136 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
ALL_BINS,
#line 136 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
SPECIFIC_BINS};
#line 137 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum BIN_ORGANIZATION {
#line 137 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
BY_BIN,
#line 137 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
BY_HISTORY};
#line 138 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum BIN_ANALYSIS_OF {
#line 138 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
WEPLS,
#line 138 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
ANGLES,
#line 138 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
POSITIONS,
#line 138 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
BIN_NUMS};
#line 187 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum FILTER_TYPES {
#line 187 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
RAM_LAK,
#line 187 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
SHEPP_LOGAN,
#line 187 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
NONE};
#line 226 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum HULL_TYPES {
#line 226 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
SC_HULL,
#line 226 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
MSC_HULL,
#line 226 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
SM_HULL,
#line 226 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
FBP_HULL};
#line 242 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
enum INITIAL_ITERATE {
#line 242 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
X_HULL,
#line 242 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
FBP_IMAGE,
#line 242 "c:\\users\\blake\\documents\\github\\pct-reconstruction\\pCT_Reconstruction.h"
HYBRID};
#line 93 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\exception"
struct _ZSt9exception;
#line 2620 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\memory"
enum _ZNSt14pointer_safety14pointer_safetyE {
#line 2621 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\memory"
_ZNSt14pointer_safety7relaxedE,
#line 2622 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\memory"
_ZNSt14pointer_safety9preferredE,
#line 2623 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\memory"
_ZNSt14pointer_safety6strictE};
#line 22 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
enum _ZNSt4errc4errcE {
#line 23 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc28address_family_not_supportedE = 102,
#line 24 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14address_in_useE = 100,
#line 25 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc21address_not_availableE,
#line 26 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc17already_connectedE = 113,
#line 27 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc22argument_list_too_longE = 7,
#line 28 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc22argument_out_of_domainE = 33,
#line 29 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc11bad_addressE = 14,
#line 30 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19bad_file_descriptorE = 9,
#line 31 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc11bad_messageE = 104,
#line 32 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc11broken_pipeE = 32,
#line 33 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc18connection_abortedE = 106,
#line 34 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc30connection_already_in_progressE = 103,
#line 35 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc18connection_refusedE = 107,
#line 36 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc16connection_resetE,
#line 37 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc17cross_device_linkE = 18,
#line 38 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc28destination_address_requiredE = 109,
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc23device_or_resource_busyE = 16,
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19directory_not_emptyE = 41,
#line 41 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc23executable_format_errorE = 8,
#line 42 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc11file_existsE = 17,
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14file_too_largeE = 27,
#line 44 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc17filename_too_longE = 38,
#line 45 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc22function_not_supportedE = 40,
#line 46 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc16host_unreachableE = 110,
#line 47 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc18identifier_removedE,
#line 48 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc21illegal_byte_sequenceE = 42,
#line 49 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc34inappropriate_io_control_operationE = 25,
#line 50 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc11interruptedE = 4,
#line 51 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc16invalid_argumentE = 22,
#line 52 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc12invalid_seekE = 29,
#line 53 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc8io_errorE = 5,
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14is_a_directoryE = 21,
#line 55 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc12message_sizeE = 115,
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc12network_downE,
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc13network_resetE,
#line 58 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19network_unreachableE,
#line 59 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc15no_buffer_spaceE,
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc16no_child_processE = 10,
#line 61 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc7no_linkE = 121,
#line 62 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc17no_lock_availableE = 39,
#line 63 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc20no_message_availableE = 120,
#line 64 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc10no_messageE = 122,
#line 65 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc18no_protocol_optionE,
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc18no_space_on_deviceE = 28,
#line 67 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19no_stream_resourcesE = 124,
#line 68 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc25no_such_device_or_addressE = 6,
#line 69 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14no_such_deviceE = 19,
#line 70 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc25no_such_file_or_directoryE = 2,
#line 71 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc15no_such_processE,
#line 72 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc15not_a_directoryE = 20,
#line 73 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc12not_a_socketE = 128,
#line 74 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc12not_a_streamE = 125,
#line 75 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc13not_connectedE,
#line 76 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc17not_enough_memoryE = 12,
#line 77 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc13not_supportedE = 129,
#line 78 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc18operation_canceledE = 105,
#line 79 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc21operation_in_progressE = 112,
#line 80 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc23operation_not_permittedE = 1,
#line 81 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc23operation_not_supportedE = 130,
#line 82 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc21operation_would_blockE = 140,
#line 83 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc10owner_deadE = 133,
#line 84 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc17permission_deniedE = 13,
#line 85 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14protocol_errorE = 134,
#line 86 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc22protocol_not_supportedE,
#line 87 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc21read_only_file_systemE = 30,
#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc29resource_deadlock_would_occurE = 36,
#line 89 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc30resource_unavailable_try_againE = 11,
#line 90 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19result_out_of_rangeE = 34,
#line 91 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc21state_not_recoverableE = 127,
#line 92 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14stream_timeoutE = 137,
#line 93 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14text_file_busyE = 139,
#line 94 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc9timed_outE = 138,
#line 95 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc29too_many_files_open_in_systemE = 23,
#line 96 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19too_many_files_openE,
#line 97 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc14too_many_linksE = 31,
#line 98 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc29too_many_synbolic_link_levelsE = 114,
#line 99 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc15value_too_largeE = 132,
#line 100 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt4errc19wrong_protocol_typeE = 136};
#line 128 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
enum _ZNSt7io_errc7io_errcE {
#line 129 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt7io_errc6streamE = 1};
#line 592 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
enum _ZSt14_Uninitialized {
#line 594 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
_ZSt7_Noinit};
#line 601 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
struct _ZSt7_Lockit;
#line 742 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
struct _ZSt6_Mutex;
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt12_Bool_struct;
#line 340 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\exception"
struct _ZSt9bad_alloc;
#line 474 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt11char_traitsIcE;
#line 31 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
enum _ZSt18float_denorm_style {
#line 32 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt20denorm_indeterminate = (-1),
#line 33 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt13denorm_absent,
#line 34 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt14denorm_present};
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
enum _ZSt17float_round_style {
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt19round_indeterminate = (-1),
#line 41 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt17round_toward_zero,
#line 42 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt16round_to_nearest,
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt21round_toward_infinity,
#line 44 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\limits"
_ZSt25round_toward_neg_infinity};
#line 34 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt16_Container_base0;
#line 45 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt15_Iterator_base0;
#line 295 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt18input_iterator_tag;
#line 299 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt19output_iterator_tag;
#line 313 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt26random_access_iterator_tag;
#line 323 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt27_Nonscalar_ptr_iterator_tag;
#line 336 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt8iteratorISt19output_iterator_tagvvvvE;
#line 363 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt6_Outit;
#line 92 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\typeinfo"
struct _ZSt8bad_cast;
#line 116 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSt15_Allocator_baseIcE;
#line 569 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSaIcE;
#line 494 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
enum _ZNSt11_String_valIcSaIcEEUt_E {
#line 495 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIcSaIcEE9_BUF_SIZEE = 16};
#line 498 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
enum _ZNSt11_String_valIcSaIcEEUt0_E {
#line 499 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIcSaIcEE11_ALLOC_MASKE = 15};
#line 504 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
union _ZNSt11_String_valIcSaIcEE5_BxtyE;
#line 445 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSt11_String_valIcSaIcEE;
#line 520 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSs;
#line 157 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdexcept"
struct _ZSt13runtime_error;
#line 3308 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt5_YarnIcE;
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt8_Locinfo;
#line 24 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt8_LocbaseIiE;
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale2idE;
#line 98 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale5facetE;
#line 192 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale7_LocimpE;
#line 59 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt6locale;
#line 746 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
enum _ZNSt12codecvt_baseUt_E {
#line 747 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base2okE,
#line 747 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base7partialE,
#line 747 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base5errorE,
#line 747 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base6noconvE};
#line 741 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt12codecvt_base;
#line 2001 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
enum _ZNSt10ctype_baseUt_E {
#line 2002 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5alnumE = 263,
#line 2002 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5alphaE = 259,
#line 2003 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5cntrlE = 32,
#line 2003 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5digitE = 4,
#line 2003 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5graphE = 279,
#line 2004 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5lowerE = 2,
#line 2004 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5printE = 471,
#line 2005 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5punctE = 16,
#line 2005 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5spaceE = 72,
#line 2005 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5upperE = 1,
#line 2006 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt10ctype_base6xdigitE = 128};
#line 1997 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt10ctype_base;
#line 2261 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt5ctypeIcE;
#line 797 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt7codecvtIcciE;
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
struct _ZSt14error_category;
#line 191 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
struct _ZSt10error_code;
#line 502 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
struct _ZSt12system_error;
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE11_Dummy_enumE {
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE15_Dummy_enum_valE = 1};
#line 55 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE9_FmtflagsE {
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE8_FmtmaskE = 65535,
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE8_FmtzeroE = 0};
#line 86 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE8_IostateE {
#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_StatmaskE = 23};
#line 96 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE9_OpenmodeE {
#line 98 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_OpenmaskE = 255};
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE8_SeekdirE {
#line 111 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_SeekmaskE = 3};
#line 118 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiEUt_E {
#line 119 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_OpenprotE = 64};
#line 51 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZSt5_IosbIiE;
#line 214 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
enum _ZNSt8ios_base5eventE {
#line 216 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt8ios_base11erase_eventE,
#line 216 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt8ios_base11imbue_eventE,
#line 216 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt8ios_base13copyfmt_eventE};
#line 222 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZNSt8ios_base7failureE;
#line 569 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZNSt8ios_base9_IosarrayE;
#line 584 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZNSt8ios_base8_FnarrayE;
#line 202 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZSt8ios_base;
#line 582 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt15basic_streambufIcSt11char_traitsIcEE;
#line 624 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE;
#line 627 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE;
#line 573 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt9basic_iosIcSt11char_traitsIcEE;
#line 86 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
struct _ZNSo12_Sentry_baseE;
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
struct _ZNSo6sentryE;
#line 588 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSo;
#line 588 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct __SO__So;
#line 71 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
struct _ZNSi12_Sentry_baseE;
#line 93 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
struct _ZNSi6sentryE;
#line 585 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSi;
#line 585 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct __SO__Si;
#line 494 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
enum _ZNSt11_String_valIwSaIwEEUt_E {
#line 495 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIwSaIwEE9_BUF_SIZEE = 8};
#line 498 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
enum _ZNSt11_String_valIwSaIwEEUt0_E {
#line 499 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIwSaIwEE11_ALLOC_MASKE = 7};
#line 116 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSt15_Allocator_baseIiE;
#line 130 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSaIiE;
#line 417 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt11_Vector_valIiSaIiEE;
#line 479 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt6vectorIiSaIiEE;
#line 116 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSt15_Allocator_baseIfE;
#line 130 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSaIfE;
#line 417 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt11_Vector_valIfSaIfEE;
#line 479 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt6vectorIfSaIfEE;
#line 206 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
enum _ZNSt13basic_filebufIcSt11char_traitsIcEE7_InitflE {
#line 208 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
_ZNSt13basic_filebufIcSt11char_traitsIcEE6_NewflE,
#line 208 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
_ZNSt13basic_filebufIcSt11char_traitsIcEE7_OpenflE,
#line 208 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
_ZNSt13basic_filebufIcSt11char_traitsIcEE8_CloseflE};
#line 610 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt13basic_filebufIcSt11char_traitsIcEE;
#line 616 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt14basic_ofstreamIcSt11char_traitsIcEE;
#line 613 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt14basic_ifstreamIcSt11char_traitsIcEE;
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIffbE;
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt7greaterIfE;
#line 134 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt14unary_functionIfbE;
#line 319 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt9binder2ndISt7greaterIfEE;
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIiibE;
#line 164 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt10logical_orIiE;
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIfffE;
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt10multipliesIfE;
#line 352 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt12_Iterator012ISt26random_access_iterator_tagfxPKfRS1_St15_Iterator_base0E;
#line 20 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE;
#line 285 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE;
#line 352 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt12_Iterator012ISt26random_access_iterator_tagixPKiRS1_St15_Iterator_base0E;
#line 20 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE;
#line 285 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE;
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIiiiE;
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt7dividesIiE;
#line 134 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt14unary_functionIiiE;
#line 319 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt9binder2ndISt7dividesIiEE;
#line 352 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt12_Iterator012ISt26random_access_iterator_tagcxPKcRS1_St15_Iterator_base0E;
#line 27 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSt22_String_const_iteratorIcSt11char_traitsIcESaIcEE;
#line 305 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSt16_String_iteratorIcSt11char_traitsIcESaIcEE;
#line 579 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt19ostreambuf_iteratorIcSt11char_traitsIcEE;
#line 336 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt8iteratorISt18input_iterator_tagcxPcRcE;
#line 576 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt19istreambuf_iteratorIcSt11char_traitsIcEE;
#line 27 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
typedef unsigned long long size_t;
#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"





































#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"





















































































#line 87 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"










#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"








































#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"










#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"






#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"




#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"










#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"













        





#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"








#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"


#line 206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\host_defines.h"
#line 39 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"





typedef __declspec(__device_builtin_texture_type__) unsigned long long __texture_type__;
typedef __declspec(__device_builtin_surface_type__) unsigned long long __surface_type__;



#line 50 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"




































































































#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"





























extern __declspec(__device__) void* malloc(size_t);
extern __declspec(__device__) void free(void*);

extern __declspec(__device__) void __assertfail(
  const void  *message,
  const void  *file,
  unsigned int line,
  const void  *function,
  size_t       charsize);















#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"













#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"













#line 233 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"
static __declspec(__device__) void _wassert(
  const unsigned short *_Message,
  const unsigned short *_File,
  unsigned              _Line)
{
  __assertfail(
    (const void *)_Message,
    (const void *)_File,
                  _Line,
    (const void *)0,
    sizeof(unsigned short));
}
#line 246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"

#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"

#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\builtin_types.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_types.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\host_defines.h"












































































































































































































#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\host_defines.h"
#line 54 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_types.h"







enum __declspec(__device_builtin__) cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};

#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_types.h"
#line 57 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\host_defines.h"












































































































































































































#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\host_defines.h"
#line 54 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"































































#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"










enum __declspec(__device_builtin__) cudaError
{
    




    cudaSuccess                           =      0,
  
    



    cudaErrorMissingConfiguration         =      1,
  
    



    cudaErrorMemoryAllocation             =      2,
  
    



    cudaErrorInitializationError          =      3,
  
    







    cudaErrorLaunchFailure                =      4,
  
    






    cudaErrorPriorLaunchFailure           =      5,
  
    







    cudaErrorLaunchTimeout                =      6,
  
    






    cudaErrorLaunchOutOfResources         =      7,
  
    



    cudaErrorInvalidDeviceFunction        =      8,
  
    






    cudaErrorInvalidConfiguration         =      9,
  
    



    cudaErrorInvalidDevice                =     10,
  
    



    cudaErrorInvalidValue                 =     11,
  
    



    cudaErrorInvalidPitchValue            =     12,
  
    



    cudaErrorInvalidSymbol                =     13,
  
    


    cudaErrorMapBufferObjectFailed        =     14,
  
    


    cudaErrorUnmapBufferObjectFailed      =     15,
  
    



    cudaErrorInvalidHostPointer           =     16,
  
    



    cudaErrorInvalidDevicePointer         =     17,
  
    



    cudaErrorInvalidTexture               =     18,
  
    



    cudaErrorInvalidTextureBinding        =     19,
  
    




    cudaErrorInvalidChannelDescriptor     =     20,
  
    



    cudaErrorInvalidMemcpyDirection       =     21,
  
    







    cudaErrorAddressOfConstant            =     22,
  
    






    cudaErrorTextureFetchFailed           =     23,
  
    






    cudaErrorTextureNotBound              =     24,
  
    






    cudaErrorSynchronizationError         =     25,
  
    



    cudaErrorInvalidFilterSetting         =     26,
  
    



    cudaErrorInvalidNormSetting           =     27,
  
    





    cudaErrorMixedDeviceExecution         =     28,
  
    




    cudaErrorCudartUnloading              =     29,
  
    


    cudaErrorUnknown                      =     30,

    





    cudaErrorNotYetImplemented            =     31,
  
    






    cudaErrorMemoryValueTooLarge          =     32,
  
    




    cudaErrorInvalidResourceHandle        =     33,
  
    





    cudaErrorNotReady                     =     34,
  
    




    cudaErrorInsufficientDriver           =     35,
  
    










    cudaErrorSetOnActiveProcess           =     36,
  
    



    cudaErrorInvalidSurface               =     37,
  
    



    cudaErrorNoDevice                     =     38,
  
    



    cudaErrorECCUncorrectable             =     39,
  
    


    cudaErrorSharedObjectSymbolNotFound   =     40,
  
    


    cudaErrorSharedObjectInitFailed       =     41,
  
    



    cudaErrorUnsupportedLimit             =     42,
  
    



    cudaErrorDuplicateVariableName        =     43,
  
    



    cudaErrorDuplicateTextureName         =     44,
  
    



    cudaErrorDuplicateSurfaceName         =     45,
  
    







    cudaErrorDevicesUnavailable           =     46,
  
    


    cudaErrorInvalidKernelImage           =     47,
  
    





    cudaErrorNoKernelImageForDevice       =     48,
  
    










    cudaErrorIncompatibleDriverContext    =     49,
      
    




    cudaErrorPeerAccessAlreadyEnabled     =     50,
    
    




    cudaErrorPeerAccessNotEnabled         =     51,
    
    



    cudaErrorDeviceAlreadyInUse           =     54,

    




    cudaErrorProfilerDisabled             =     55,

    





    cudaErrorProfilerNotInitialized       =     56,

    




    cudaErrorProfilerAlreadyStarted       =     57,

    




     cudaErrorProfilerAlreadyStopped       =    58,

    





    cudaErrorAssert                        =    59,
  
    




    cudaErrorTooManyPeers                 =     60,
  
    



    cudaErrorHostMemoryAlreadyRegistered  =     61,
        
    



    cudaErrorHostMemoryNotRegistered      =     62,

    


    cudaErrorOperatingSystem              =     63,

    



    cudaErrorPeerAccessUnsupported        =     64,

    




    cudaErrorLaunchMaxDepthExceeded       =     65,

    





    cudaErrorLaunchFileScopedTex          =     66,

    





    cudaErrorLaunchFileScopedSurf         =     67,

    












    cudaErrorSyncDepthExceeded            =     68,

    









    cudaErrorLaunchPendingCountExceeded   =     69,
    
    


    cudaErrorNotPermitted                 =     70,

    



    cudaErrorNotSupported                 =     71,

    






    cudaErrorHardwareStackError           =     72,

    





    cudaErrorIllegalInstruction           =     73,

    






    cudaErrorMisalignedAddress            =     74,

    








    cudaErrorInvalidAddressSpace          =     75,

    





    cudaErrorInvalidPc                    =     76,

    





    cudaErrorIllegalAddress               =     77,


    


    cudaErrorStartupFailure               =   0x7f,

    





    cudaErrorApiFailureBase               =  10000
};




enum __declspec(__device_builtin__) cudaChannelFormatKind
{
    cudaChannelFormatKindSigned           =   0,      
    cudaChannelFormatKindUnsigned         =   1,      
    cudaChannelFormatKindFloat            =   2,      
    cudaChannelFormatKindNone             =   3       
};




struct __declspec(__device_builtin__) cudaChannelFormatDesc
{
    int                        x; 
    int                        y; 
    int                        z; 
    int                        w; 
    enum cudaChannelFormatKind f; 
};




typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;




typedef struct cudaMipmappedArray *cudaMipmappedArray_t;




typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;




enum __declspec(__device_builtin__) cudaMemoryType
{
    cudaMemoryTypeHost   = 1, 
    cudaMemoryTypeDevice = 2  
};




enum __declspec(__device_builtin__) cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      
    cudaMemcpyHostToDevice        =   1,      
    cudaMemcpyDeviceToHost        =   2,      
    cudaMemcpyDeviceToDevice      =   3,      
    cudaMemcpyDefault             =   4       
};





struct __declspec(__device_builtin__) cudaPitchedPtr
{
    void   *ptr;      
    size_t  pitch;    
    size_t  xsize;    
    size_t  ysize;    
};





struct __declspec(__device_builtin__) cudaExtent
{
    size_t width;     
    size_t height;    
    size_t depth;     
};





struct __declspec(__device_builtin__) cudaPos
{
    size_t x;     
    size_t y;     
    size_t z;     
};




struct __declspec(__device_builtin__) cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  
    struct cudaPos         srcPos;    
    struct cudaPitchedPtr  srcPtr;    
  
    cudaArray_t            dstArray;  
    struct cudaPos         dstPos;    
    struct cudaPitchedPtr  dstPtr;    
  
    struct cudaExtent      extent;    
    enum cudaMemcpyKind    kind;      
};




struct __declspec(__device_builtin__) cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  
    struct cudaPos         srcPos;    
    struct cudaPitchedPtr  srcPtr;    
    int                    srcDevice; 
  
    cudaArray_t            dstArray;  
    struct cudaPos         dstPos;    
    struct cudaPitchedPtr  dstPtr;    
    int                    dstDevice; 
  
    struct cudaExtent      extent;    
};




struct cudaGraphicsResource;




enum __declspec(__device_builtin__) cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  
    cudaGraphicsRegisterFlagsReadOnly         = 1,   
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  
    cudaGraphicsRegisterFlagsTextureGather    = 8   
};




enum __declspec(__device_builtin__) cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  
    cudaGraphicsMapFlagsReadOnly     = 1,  
    cudaGraphicsMapFlagsWriteDiscard = 2   
};




enum __declspec(__device_builtin__) cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, 
    cudaGraphicsCubeFaceNegativeX = 0x01, 
    cudaGraphicsCubeFacePositiveY = 0x02, 
    cudaGraphicsCubeFaceNegativeY = 0x03, 
    cudaGraphicsCubeFacePositiveZ = 0x04, 
    cudaGraphicsCubeFaceNegativeZ = 0x05  
};




enum __declspec(__device_builtin__) cudaResourceType
{
    cudaResourceTypeArray          = 0x00, 
    cudaResourceTypeMipmappedArray = 0x01, 
    cudaResourceTypeLinear         = 0x02, 
    cudaResourceTypePitch2D        = 0x03  
};




enum __declspec(__device_builtin__) cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, 
    cudaResViewFormatUnsignedChar1             = 0x01, 
    cudaResViewFormatUnsignedChar2             = 0x02, 
    cudaResViewFormatUnsignedChar4             = 0x03, 
    cudaResViewFormatSignedChar1               = 0x04, 
    cudaResViewFormatSignedChar2               = 0x05, 
    cudaResViewFormatSignedChar4               = 0x06, 
    cudaResViewFormatUnsignedShort1            = 0x07, 
    cudaResViewFormatUnsignedShort2            = 0x08, 
    cudaResViewFormatUnsignedShort4            = 0x09, 
    cudaResViewFormatSignedShort1              = 0x0a, 
    cudaResViewFormatSignedShort2              = 0x0b, 
    cudaResViewFormatSignedShort4              = 0x0c, 
    cudaResViewFormatUnsignedInt1              = 0x0d, 
    cudaResViewFormatUnsignedInt2              = 0x0e, 
    cudaResViewFormatUnsignedInt4              = 0x0f, 
    cudaResViewFormatSignedInt1                = 0x10, 
    cudaResViewFormatSignedInt2                = 0x11, 
    cudaResViewFormatSignedInt4                = 0x12, 
    cudaResViewFormatHalf1                     = 0x13, 
    cudaResViewFormatHalf2                     = 0x14, 
    cudaResViewFormatHalf4                     = 0x15, 
    cudaResViewFormatFloat1                    = 0x16, 
    cudaResViewFormatFloat2                    = 0x17, 
    cudaResViewFormatFloat4                    = 0x18, 
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, 
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, 
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, 
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, 
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, 
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, 
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, 
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, 
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, 
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  
};




struct __declspec(__device_builtin__) cudaResourceDesc {
	enum cudaResourceType resType;             
	
	union {
		struct {
			cudaArray_t array;                 
		} array;
        struct {
            cudaMipmappedArray_t mipmap;       
        } mipmap;
		struct {
			void *devPtr;                      
			struct cudaChannelFormatDesc desc; 
			size_t sizeInBytes;                
		} linear;
		struct {
			void *devPtr;                      
			struct cudaChannelFormatDesc desc; 
			size_t width;                      
			size_t height;                     
			size_t pitchInBytes;               
		} pitch2D;
	} res;
};




struct __declspec(__device_builtin__) cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           
    size_t                      width;            
    size_t                      height;           
    size_t                      depth;            
    unsigned int                firstMipmapLevel; 
    unsigned int                lastMipmapLevel;  
    unsigned int                firstLayer;       
    unsigned int                lastLayer;        
};




struct __declspec(__device_builtin__) cudaPointerAttributes
{
    



    enum cudaMemoryType memoryType;

    








    int device;

    



    void *devicePointer;

    



    void *hostPointer;

    


    int isManaged;
};




struct __declspec(__device_builtin__) cudaFuncAttributes
{
   




   size_t sharedSizeBytes;

   



   size_t constSizeBytes;

   


   size_t localSizeBytes;

   




   int maxThreadsPerBlock;

   


   int numRegs;

   




   int ptxVersion;

   




   int binaryVersion;

   



   int cacheModeCA;
};




enum __declspec(__device_builtin__) cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    
    cudaFuncCachePreferShared = 1,    
    cudaFuncCachePreferL1     = 2,    
    cudaFuncCachePreferEqual  = 3     
};





enum __declspec(__device_builtin__) cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum __declspec(__device_builtin__) cudaComputeMode
{
    cudaComputeModeDefault          = 0,  
    cudaComputeModeExclusive        = 1,  
    cudaComputeModeProhibited       = 2,  
    cudaComputeModeExclusiveProcess = 3   
};




enum __declspec(__device_builtin__) cudaLimit
{
    cudaLimitStackSize                    = 0x00, 
    cudaLimitPrintfFifoSize               = 0x01, 
    cudaLimitMallocHeapSize               = 0x02, 
    cudaLimitDevRuntimeSyncDepth          = 0x03, 
    cudaLimitDevRuntimePendingLaunchCount = 0x04  
};




enum __declspec(__device_builtin__) cudaOutputMode
{
    cudaKeyValuePair    = 0x00, 
    cudaCSV             = 0x01  
};




enum __declspec(__device_builtin__) cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  
    cudaDevAttrMaxBlockDimX                   = 2,  
    cudaDevAttrMaxBlockDimY                   = 3,  
    cudaDevAttrMaxBlockDimZ                   = 4,  
    cudaDevAttrMaxGridDimX                    = 5,  
    cudaDevAttrMaxGridDimY                    = 6,  
    cudaDevAttrMaxGridDimZ                    = 7,  
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  
    cudaDevAttrTotalConstantMemory            = 9,  
    cudaDevAttrWarpSize                       = 10, 
    cudaDevAttrMaxPitch                       = 11, 
    cudaDevAttrMaxRegistersPerBlock           = 12, 
    cudaDevAttrClockRate                      = 13, 
    cudaDevAttrTextureAlignment               = 14, 
    cudaDevAttrGpuOverlap                     = 15, 
    cudaDevAttrMultiProcessorCount            = 16, 
    cudaDevAttrKernelExecTimeout              = 17, 
    cudaDevAttrIntegrated                     = 18, 
    cudaDevAttrCanMapHostMemory               = 19, 
    cudaDevAttrComputeMode                    = 20, 
    cudaDevAttrMaxTexture1DWidth              = 21, 
    cudaDevAttrMaxTexture2DWidth              = 22, 
    cudaDevAttrMaxTexture2DHeight             = 23, 
    cudaDevAttrMaxTexture3DWidth              = 24, 
    cudaDevAttrMaxTexture3DHeight             = 25, 
    cudaDevAttrMaxTexture3DDepth              = 26, 
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, 
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, 
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, 
    cudaDevAttrSurfaceAlignment               = 30, 
    cudaDevAttrConcurrentKernels              = 31, 
    cudaDevAttrEccEnabled                     = 32, 
    cudaDevAttrPciBusId                       = 33, 
    cudaDevAttrPciDeviceId                    = 34, 
    cudaDevAttrTccDriver                      = 35, 
    cudaDevAttrMemoryClockRate                = 36, 
    cudaDevAttrGlobalMemoryBusWidth           = 37, 
    cudaDevAttrL2CacheSize                    = 38, 
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, 
    cudaDevAttrAsyncEngineCount               = 40, 
    cudaDevAttrUnifiedAddressing              = 41,     
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, 
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, 
    cudaDevAttrMaxTexture2DGatherWidth        = 45, 
    cudaDevAttrMaxTexture2DGatherHeight       = 46, 
    cudaDevAttrMaxTexture3DWidthAlt           = 47, 
    cudaDevAttrMaxTexture3DHeightAlt          = 48, 
    cudaDevAttrMaxTexture3DDepthAlt           = 49, 
    cudaDevAttrPciDomainId                    = 50, 
    cudaDevAttrTexturePitchAlignment          = 51, 
    cudaDevAttrMaxTextureCubemapWidth         = 52, 
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, 
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, 
    cudaDevAttrMaxSurface1DWidth              = 55, 
    cudaDevAttrMaxSurface2DWidth              = 56, 
    cudaDevAttrMaxSurface2DHeight             = 57, 
    cudaDevAttrMaxSurface3DWidth              = 58, 
    cudaDevAttrMaxSurface3DHeight             = 59, 
    cudaDevAttrMaxSurface3DDepth              = 60, 
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, 
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, 
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, 
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, 
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, 
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, 
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, 
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, 
    cudaDevAttrMaxTexture1DLinearWidth        = 69, 
    cudaDevAttrMaxTexture2DLinearWidth        = 70, 
    cudaDevAttrMaxTexture2DLinearHeight       = 71, 
    cudaDevAttrMaxTexture2DLinearPitch        = 72, 
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, 
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, 
    cudaDevAttrComputeCapabilityMajor         = 75,  
    cudaDevAttrComputeCapabilityMinor         = 76, 
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, 
    cudaDevAttrStreamPrioritiesSupported      = 78, 
    cudaDevAttrGlobalL1CacheSupported         = 79, 
    cudaDevAttrLocalL1CacheSupported          = 80, 
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, 
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, 
    cudaDevAttrManagedMemory                  = 83, 
    cudaDevAttrIsMultiGpuBoard                = 84, 
    cudaDevAttrMultiGpuBoardGroupID           = 85  
};




struct __declspec(__device_builtin__) cudaDeviceProp
{
    char   name[256];                  
    size_t totalGlobalMem;             
    size_t sharedMemPerBlock;          
    int    regsPerBlock;               
    int    warpSize;                   
    size_t memPitch;                   
    int    maxThreadsPerBlock;         
    int    maxThreadsDim[3];           
    int    maxGridSize[3];             
    int    clockRate;                  
    size_t totalConstMem;              
    int    major;                      
    int    minor;                      
    size_t textureAlignment;           
    size_t texturePitchAlignment;      
    int    deviceOverlap;              
    int    multiProcessorCount;        
    int    kernelExecTimeoutEnabled;   
    int    integrated;                 
    int    canMapHostMemory;           
    int    computeMode;                
    int    maxTexture1D;               
    int    maxTexture1DMipmap;         
    int    maxTexture1DLinear;         
    int    maxTexture2D[2];            
    int    maxTexture2DMipmap[2];      
    int    maxTexture2DLinear[3];      
    int    maxTexture2DGather[2];      
    int    maxTexture3D[3];            
    int    maxTexture3DAlt[3];         
    int    maxTextureCubemap;          
    int    maxTexture1DLayered[2];     
    int    maxTexture2DLayered[3];     
    int    maxTextureCubemapLayered[2];
    int    maxSurface1D;               
    int    maxSurface2D[2];            
    int    maxSurface3D[3];            
    int    maxSurface1DLayered[2];     
    int    maxSurface2DLayered[3];     
    int    maxSurfaceCubemap;          
    int    maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;           
    int    concurrentKernels;          
    int    ECCEnabled;                 
    int    pciBusID;                   
    int    pciDeviceID;                
    int    pciDomainID;                
    int    tccDriver;                  
    int    asyncEngineCount;           
    int    unifiedAddressing;          
    int    memoryClockRate;            
    int    memoryBusWidth;             
    int    l2CacheSize;                
    int    maxThreadsPerMultiProcessor;
    int    streamPrioritiesSupported;  
    int    globalL1CacheSupported;     
    int    localL1CacheSupported;      
    size_t sharedMemPerMultiprocessor; 
    int    regsPerMultiprocessor;      
    int    managedMemory;              
    int    isMultiGpuBoard;            
    int    multiGpuBoardGroupID;       
};











































































typedef __declspec(__device_builtin__) struct __declspec(__device_builtin__) cudaIpcEventHandle_st
{
    char reserved[64];
}cudaIpcEventHandle_t;




typedef __declspec(__device_builtin__) struct __declspec(__device_builtin__) cudaIpcMemHandle_st 
{
    char reserved[64];
}cudaIpcMemHandle_t;










typedef __declspec(__device_builtin__) enum cudaError cudaError_t;




typedef __declspec(__device_builtin__) struct CUstream_st *cudaStream_t;




typedef __declspec(__device_builtin__) struct CUevent_st *cudaEvent_t;




typedef __declspec(__device_builtin__) struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef __declspec(__device_builtin__) struct CUuuid_st cudaUUID_t;




typedef __declspec(__device_builtin__) enum cudaOutputMode cudaOutputMode_t;


 

#line 1414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"

#line 58 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\surface_types.h"


























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"




































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\surface_types.h"
























enum __declspec(__device_builtin__) cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero  = 0,    
    cudaBoundaryModeClamp = 1,    
    cudaBoundaryModeTrap  = 2     
};




enum __declspec(__device_builtin__)  cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,     
    cudaFormatModeAuto = 1        
};




struct __declspec(__device_builtin__) surfaceReference
{
    


    struct cudaChannelFormatDesc channelDesc;
};




typedef __declspec(__device_builtin__) unsigned long long cudaSurfaceObject_t;


 

#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\surface_types.h"
#line 59 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\texture_types.h"


























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"




































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\texture_types.h"
























enum __declspec(__device_builtin__) cudaTextureAddressMode
{
    cudaAddressModeWrap   = 0,    
    cudaAddressModeClamp  = 1,    
    cudaAddressModeMirror = 2,    
    cudaAddressModeBorder = 3     
};




enum __declspec(__device_builtin__) cudaTextureFilterMode
{
    cudaFilterModePoint  = 0,     
    cudaFilterModeLinear = 1      
};




enum __declspec(__device_builtin__) cudaTextureReadMode
{
    cudaReadModeElementType     = 0,  
    cudaReadModeNormalizedFloat = 1   
};




struct __declspec(__device_builtin__) textureReference
{
    


    int                          normalized;
    


    enum cudaTextureFilterMode   filterMode;
    


    enum cudaTextureAddressMode  addressMode[3];
    


    struct cudaChannelFormatDesc channelDesc;
    


    int                          sRGB;
    


    unsigned int                 maxAnisotropy;
    


    enum cudaTextureFilterMode   mipmapFilterMode;
    


    float                        mipmapLevelBias;
    


    float                        minMipmapLevelClamp;
    


    float                        maxMipmapLevelClamp;
    int                          __cudaReserved[15];
};




struct __declspec(__device_builtin__) cudaTextureDesc
{
    


    enum cudaTextureAddressMode addressMode[3];
    


    enum cudaTextureFilterMode  filterMode;
    


    enum cudaTextureReadMode    readMode;
    


    int                         sRGB;
    


    int                         normalizedCoords;
    


    unsigned int                maxAnisotropy;
    


    enum cudaTextureFilterMode  mipmapFilterMode;
    


    float                       mipmapLevelBias;
    


    float                       minMipmapLevelClamp;
    


    float                       maxMipmapLevelClamp;
};




typedef __declspec(__device_builtin__) unsigned long long cudaTextureObject_t;


 

#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\texture_types.h"
#line 60 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"



























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_types.h"




































































#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_types.h"
#line 57 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"




































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\driver_types.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\surface_types.h"






















































































































#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\surface_types.h"
#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\texture_types.h"




















































































































































































































#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\texture_types.h"
#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"












































































































































































































































































































































































































































#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"
#line 62 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\host_defines.h"












































































































































































































#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\host_defines.h"
#line 63 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"






















#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"







#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"

struct __declspec(__device_builtin__) char1
{
    signed char x;
};

struct __declspec(__device_builtin__) uchar1
{
    unsigned char x;
};


struct __declspec(__device_builtin__) __declspec(align(2)) char2
{
    signed char x, y;
};

struct __declspec(__device_builtin__) __declspec(align(2)) uchar2
{
    unsigned char x, y;
};

struct __declspec(__device_builtin__) char3
{
    signed char x, y, z;
};

struct __declspec(__device_builtin__) uchar3
{
    unsigned char x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(4)) char4
{
    signed char x, y, z, w;
};

struct __declspec(__device_builtin__) __declspec(align(4)) uchar4
{
    unsigned char x, y, z, w;
};

struct __declspec(__device_builtin__) short1
{
    short x;
};

struct __declspec(__device_builtin__) ushort1
{
    unsigned short x;
};

struct __declspec(__device_builtin__) __declspec(align(4)) short2
{
    short x, y;
};

struct __declspec(__device_builtin__) __declspec(align(4)) ushort2
{
    unsigned short x, y;
};

struct __declspec(__device_builtin__) short3
{
    short x, y, z;
};

struct __declspec(__device_builtin__) ushort3
{
    unsigned short x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(8)) short4 { short x; short y; short z; short w; };
struct __declspec(__device_builtin__) __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct __declspec(__device_builtin__) int1
{
    int x;
};

struct __declspec(__device_builtin__) uint1
{
    unsigned int x;
};

struct __declspec(__device_builtin__) __declspec(align(8)) int2 { int x; int y; };
struct __declspec(__device_builtin__) __declspec(align(8)) uint2 { unsigned int x; unsigned int y; };

struct __declspec(__device_builtin__) int3
{
    int x, y, z;
};

struct __declspec(__device_builtin__) uint3
{
    unsigned int x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) int4
{
    int x, y, z, w;
};

struct __declspec(__device_builtin__) __declspec(align(16)) uint4
{
    unsigned int x, y, z, w;
};

struct __declspec(__device_builtin__) long1
{
    long int x;
};

struct __declspec(__device_builtin__) ulong1
{
    unsigned long x;
};


struct __declspec(__device_builtin__) __declspec(align(8)) long2 { long int x; long int y; };
struct __declspec(__device_builtin__) __declspec(align(8)) ulong2 { unsigned long int x; unsigned long int y; };












#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"

struct __declspec(__device_builtin__) long3
{
    long int x, y, z;
};

struct __declspec(__device_builtin__) ulong3
{
    unsigned long int x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) long4
{
    long int x, y, z, w;
};

struct __declspec(__device_builtin__) __declspec(align(16)) ulong4
{
    unsigned long int x, y, z, w;
};

struct __declspec(__device_builtin__) float1
{
    float x;
};















#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"

struct __declspec(__device_builtin__) __declspec(align(8)) float2 { float x; float y; };

#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"


struct __declspec(__device_builtin__) float3
{
    float x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) float4
{
    float x, y, z, w;
};

struct __declspec(__device_builtin__) longlong1
{
    long long int x;
};

struct __declspec(__device_builtin__) ulonglong1
{
    unsigned long long int x;
};

struct __declspec(__device_builtin__) __declspec(align(16)) longlong2
{
    long long int x, y;
};

struct __declspec(__device_builtin__) __declspec(align(16)) ulonglong2
{
    unsigned long long int x, y;
};

struct __declspec(__device_builtin__) longlong3
{
    long long int x, y, z;
};

struct __declspec(__device_builtin__) ulonglong3
{
    unsigned long long int x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) longlong4
{
    long long int x, y, z ,w;
};

struct __declspec(__device_builtin__) __declspec(align(16)) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __declspec(__device_builtin__) double1
{
    double x;
};

struct __declspec(__device_builtin__) __declspec(align(16)) double2
{
    double x, y;
};

struct __declspec(__device_builtin__) double3
{
    double x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) double4
{
    double x, y, z, w;
};





#line 353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"







typedef __declspec(__device_builtin__) struct char1 char1;
typedef __declspec(__device_builtin__) struct uchar1 uchar1;
typedef __declspec(__device_builtin__) struct char2 char2;
typedef __declspec(__device_builtin__) struct uchar2 uchar2;
typedef __declspec(__device_builtin__) struct char3 char3;
typedef __declspec(__device_builtin__) struct uchar3 uchar3;
typedef __declspec(__device_builtin__) struct char4 char4;
typedef __declspec(__device_builtin__) struct uchar4 uchar4;
typedef __declspec(__device_builtin__) struct short1 short1;
typedef __declspec(__device_builtin__) struct ushort1 ushort1;
typedef __declspec(__device_builtin__) struct short2 short2;
typedef __declspec(__device_builtin__) struct ushort2 ushort2;
typedef __declspec(__device_builtin__) struct short3 short3;
typedef __declspec(__device_builtin__) struct ushort3 ushort3;
typedef __declspec(__device_builtin__) struct short4 short4;
typedef __declspec(__device_builtin__) struct ushort4 ushort4;
typedef __declspec(__device_builtin__) struct int1 int1;
typedef __declspec(__device_builtin__) struct uint1 uint1;
typedef __declspec(__device_builtin__) struct int2 int2;
typedef __declspec(__device_builtin__) struct uint2 uint2;
typedef __declspec(__device_builtin__) struct int3 int3;
typedef __declspec(__device_builtin__) struct uint3 uint3;
typedef __declspec(__device_builtin__) struct int4 int4;
typedef __declspec(__device_builtin__) struct uint4 uint4;
typedef __declspec(__device_builtin__) struct long1 long1;
typedef __declspec(__device_builtin__) struct ulong1 ulong1;
typedef __declspec(__device_builtin__) struct long2 long2;
typedef __declspec(__device_builtin__) struct ulong2 ulong2;
typedef __declspec(__device_builtin__) struct long3 long3;
typedef __declspec(__device_builtin__) struct ulong3 ulong3;
typedef __declspec(__device_builtin__) struct long4 long4;
typedef __declspec(__device_builtin__) struct ulong4 ulong4;
typedef __declspec(__device_builtin__) struct float1 float1;
typedef __declspec(__device_builtin__) struct float2 float2;
typedef __declspec(__device_builtin__) struct float3 float3;
typedef __declspec(__device_builtin__) struct float4 float4;
typedef __declspec(__device_builtin__) struct longlong1 longlong1;
typedef __declspec(__device_builtin__) struct ulonglong1 ulonglong1;
typedef __declspec(__device_builtin__) struct longlong2 longlong2;
typedef __declspec(__device_builtin__) struct ulonglong2 ulonglong2;
typedef __declspec(__device_builtin__) struct longlong3 longlong3;
typedef __declspec(__device_builtin__) struct ulonglong3 ulonglong3;
typedef __declspec(__device_builtin__) struct longlong4 longlong4;
typedef __declspec(__device_builtin__) struct ulonglong4 ulonglong4;
typedef __declspec(__device_builtin__) struct double1 double1;
typedef __declspec(__device_builtin__) struct double2 double2;
typedef __declspec(__device_builtin__) struct double3 double3;
typedef __declspec(__device_builtin__) struct double4 double4;







struct __declspec(__device_builtin__) dim3
{
    unsigned int x, y, z;




#line 423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"
};

typedef __declspec(__device_builtin__) struct dim3 dim3;



#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\builtin_types.h"
#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"
#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"












































































































































































































































































































































































































































#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\vector_types.h"
#line 54 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"






#line 61 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"



#line 65 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"

uint3 __declspec(__device_builtin__) extern const threadIdx;
uint3 __declspec(__device_builtin__) extern const blockIdx;
dim3 __declspec(__device_builtin__) extern const blockDim;
dim3 __declspec(__device_builtin__) extern const gridDim;
int __declspec(__device_builtin__) extern const warpSize;





#line 77 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"






#line 84 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"






#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"






#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"






#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"






#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"

#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\device_launch_parameters.h"
#line 251 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"










































#line 44 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"






#line 51 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 55 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 63 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 67 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 71 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 75 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"



#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\crt\\storage_class.h"
#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\crt/device_runtime.h"
#line 29 "c:\\program files (x86)\\microsoft visual studio 10.0\\vc\\include\\codeanalysis\\sourceannotations.h"
struct __C3 { struct __C2 *regions; void **obj_table; struct __C1 *array_table; unsigned short saved_region_number;char __nv_no_debug_dummy_end_padding_0[6];}; struct __C4 { const struct __type_info *tinfo; unsigned char flags; unsigned char *ptr_flags;}; struct __C5 { int setjmp_buffer[16]; struct __C4 *catch_entries; void *rtinfo; unsigned short region_number;char __nv_no_debug_dummy_end_padding_0[6];}; union __C6 { struct __C5 try_block; struct __C3 function; struct __C4 *throw_spec;}; struct __C7 { struct __C7 *next; unsigned char kind; union __C6 variant;}; struct __type_info { const long long *__vptr; const char *__name;}; struct __fundamental_type_info { struct __type_info base;}; struct __class_type_info { struct __type_info base;}; struct __si_class_type_info { struct __class_type_info base; const struct __class_type_info *base_type;};
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vadefs.h"
typedef char *va_list;
#line 434 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\crtdefs.h"
typedef long long ptrdiff_t;
#line 82 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\time.h"
typedef long clock_t;
#line 553 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
typedef long long _Longlong;
#line 851 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
typedef int _Mbstatet;
#pragma pack(8)
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
struct _iobuf {
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
char *_ptr;
#line 58 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
int _cnt;
#line 59 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
char *_base;
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
int _flag;
#line 61 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
int _file;
#line 62 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
int _charbuf;
#line 63 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
int _bufsiz;
#line 64 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
char *_tmpfname;};
#pragma pack()
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
typedef struct _iobuf FILE;
#pragma pack(8)
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\typeinfo"
struct type_info { const long long *__vptr;
#line 70 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\typeinfo"
void *_M_data;
#line 71 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\typeinfo"
char _M_d_name[1];char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#pragma pack(8)
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
struct _Ctypevec {
#line 59 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
unsigned long _Hand;
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
unsigned _Page;
#line 61 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
const short *_Table;
#line 62 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
int _Delfl;char __nv_no_debug_dummy_end_padding_0[4];};
#pragma pack()
#pragma pack(8)
#line 65 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
struct _Cvtvec {
#line 67 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
unsigned long _Hand;
#line 68 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo.h"
unsigned _Page;};
#pragma pack()
#pragma pack(8)
#line 93 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\exception"
struct _ZSt9exception { const long long *__vptr;
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\exception"
const char *_Mywhat;
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\exception"
__nv_bool _Mydofree;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#line 207 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
typedef int _ZNSt8ios_base7iostateE;
#line 20 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef _Longlong _ZSt10streamsize;
#line 118 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef int _ZNSt15_Allocator_baseIiE10value_typeE;
#line 135 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSt15_Allocator_baseIiE10value_typeE _ZNSaIiE10value_typeE;
#line 137 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSaIiE10value_typeE *_ZNSaIiE7pointerE;
#line 464 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE7pointerE _ZNSt11_Vector_valIiSaIiEE7pointerE;
#line 351 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xtr1common"
typedef _ZNSt11_Vector_valIiSaIiEE7pointerE _ZNSt3tr117_Remove_referenceIRPiE5_TypeE;
#line 351 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xtr1common"
typedef struct _ZSt6vectorIiSaIiEE _ZNSt3tr117_Remove_referenceIRSt6vectorIiSaIiEEE5_TypeE;
#line 118 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef float _ZNSt15_Allocator_baseIfE10value_typeE;
#line 135 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSt15_Allocator_baseIfE10value_typeE _ZNSaIfE10value_typeE;
#line 137 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSaIfE10value_typeE *_ZNSaIfE7pointerE;
#line 464 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE7pointerE _ZNSt11_Vector_valIfSaIfEE7pointerE;
#line 351 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xtr1common"
typedef _ZNSt11_Vector_valIfSaIfEE7pointerE _ZNSt3tr117_Remove_referenceIRPfE5_TypeE;
#line 351 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xtr1common"
typedef struct _ZSt6vectorIfSaIfEE _ZNSt3tr117_Remove_referenceIRSt6vectorIfSaIfEEE5_TypeE;
#line 536 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
typedef __nv_bool _ZSt5_Bool;
#pragma pack(8)
#line 601 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
struct _ZSt7_Lockit {
#line 640 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
int _Locktype;};
#pragma pack()
#pragma pack(8)
#line 742 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
struct _ZSt6_Mutex {
#line 791 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\yvals.h"
void *_Mtx;};
#pragma pack()
#pragma pack(8)
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt12_Bool_struct {
#line 110 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
int _Member;};
#pragma pack()
#line 125 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef long long _ZSt10_Bool_type;
#pragma pack(8)
#line 340 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\exception"
struct _ZSt9bad_alloc { struct _ZSt9exception __b_St9exception;};
#pragma pack()
#line 19 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef _Longlong _ZSt9streamoff;
#line 476 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef char _ZNSt11char_traitsIcE5_ElemE;
#line 478 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef int _ZNSt11char_traitsIcE8int_typeE;
#line 480 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef _ZSt9streamoff _ZNSt11char_traitsIcE8off_typeE;
#line 481 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef _Mbstatet _ZNSt11char_traitsIcE10state_typeE;
#pragma pack(8)
#line 474 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt11char_traitsIcE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 636 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef struct _ZSo _ZSt7ostream;
#line 647 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef struct _ZSt14basic_ifstreamIcSt11char_traitsIcEE _ZSt8ifstream;
#line 648 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
typedef struct _ZSt14basic_ofstreamIcSt11char_traitsIcEE _ZSt8ofstream;
#pragma pack(8)
#line 34 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt16_Container_base0 {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 45 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt15_Iterator_base0 {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 225 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
typedef struct _ZSt16_Container_base0 _ZSt15_Container_base;
#line 226 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
typedef struct _ZSt15_Iterator_base0 _ZSt14_Iterator_base;
#pragma pack(8)
#line 323 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt27_Nonscalar_ptr_iterator_tag {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 336 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt8iteratorISt19output_iterator_tagvvvvE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 363 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt6_Outit {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 92 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\typeinfo"
struct _ZSt8bad_cast { struct _ZSt9exception __b_St9exception;};
#pragma pack()
#line 2062 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef struct _ZSs _ZSt6string;
#line 118 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef char _ZNSt15_Allocator_baseIcE10value_typeE;
#pragma pack(8)
#line 116 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSt15_Allocator_baseIcE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 135 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSt15_Allocator_baseIcE10value_typeE _ZNSaIcE10value_typeE;
#line 137 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSaIcE10value_typeE *_ZNSaIcE7pointerE;
#line 139 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef const _ZNSaIcE10value_typeE *_ZNSaIcE13const_pointerE;
#line 140 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef const _ZNSaIcE10value_typeE *_ZNSaIcE15const_referenceE;
#line 142 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef size_t _ZNSaIcE9size_typeE;
#line 143 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef ptrdiff_t _ZNSaIcE15difference_typeE;
#line 148 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef struct _ZSaIcE _ZNSaIcE6rebindIcE5otherE;
#pragma pack(8)
#line 569 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSaIcE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 450 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSaIcE6rebindIcE5otherE _ZNSt11_String_valIcSaIcEE5_AltyE;
#line 485 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSaIcE9size_typeE _ZNSt11_String_valIcSaIcEE9size_typeE;
#pragma pack(8)
#line 504 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
union _ZNSt11_String_valIcSaIcEE5_BxtyE {
#line 506 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
char _Buf[16];
#line 507 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
char *_Ptr;
#line 508 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
char _Alias[16];};
#pragma pack()
#pragma pack(8)
#line 445 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSt11_String_valIcSaIcEE {
#line 509 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
union _ZNSt11_String_valIcSaIcEE5_BxtyE _Bx;
#line 511 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIcSaIcEE9size_typeE _Mysize;
#line 512 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIcSaIcEE9size_typeE _Myres;
#line 513 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt11_String_valIcSaIcEE5_AltyE _Alval;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#line 526 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSt11_String_valIcSaIcEE5_AltyE _ZNSs6_AllocE;
#line 527 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSaIcE9size_typeE _ZNSs9size_typeE;
#line 529 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSaIcE7pointerE _ZNSs7pointerE;
#line 535 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef struct _ZSt16_String_iteratorIcSt11char_traitsIcESaIcEE _ZNSs8iteratorE;
#pragma pack(8)
#line 520 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSs { struct _ZSt11_String_valIcSaIcEE __b_St11_String_valIcSaIcEE;};
#pragma pack()
#pragma pack(8)
#line 157 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdexcept"
struct _ZSt13runtime_error { struct _ZSt9exception __b_St9exception;};
#pragma pack()
#pragma pack(8)
#line 3308 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt5_YarnIcE {
#line 3404 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
char *_Myptr;
#line 3405 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
char _Nul;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
typedef struct _Ctypevec _ZNSt8_Locinfo9_CtypevecE;
#line 61 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
typedef struct _Cvtvec _ZNSt8_Locinfo7_CvtvecE;
#pragma pack(8)
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt8_Locinfo {
#line 189 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt7_Lockit _Lock;
#line 192 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt5_YarnIcE _Days;
#line 193 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt5_YarnIcE _Months;
#line 194 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt5_YarnIcE _Oldlocname;
#line 195 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocinfo"
struct _ZSt5_YarnIcE _Newlocname;};
#pragma pack()
#pragma pack(8)
#line 24 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt8_LocbaseIiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 63 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
typedef int _ZNSt6locale8categoryE;
#pragma pack(8)
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale2idE {
#line 87 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
size_t _Id;};
#pragma pack()
#pragma pack(8)
#line 98 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale5facetE { const long long *__vptr;
#line 174 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
size_t _Refs;};
#pragma pack()
#pragma pack(8)
#line 192 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale7_LocimpE { struct _ZNSt6locale5facetE __b_NSt6locale5facetE;
#line 243 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale5facetE **_Facetvec;
#line 244 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
size_t _Facetcount;
#line 245 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt6locale8categoryE _Catmask;
#line 246 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
__nv_bool _Xparent;
#line 247 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt5_YarnIcE _Name;};
#pragma pack()
#pragma pack(8)
#line 59 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt6locale {
#line 482 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZNSt6locale7_LocimpE *_Ptr;};
#pragma pack()
#line 748 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
typedef int _ZNSt12codecvt_base6resultE;
#pragma pack(8)
#line 741 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt12codecvt_base { struct _ZNSt6locale5facetE __b_NSt6locale5facetE;};
#pragma pack()
#line 2007 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
typedef short _ZNSt10ctype_base4maskE;
#pragma pack(8)
#line 1997 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt10ctype_base { struct _ZNSt6locale5facetE __b_NSt6locale5facetE;};
#pragma pack()
#line 2264 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
typedef struct _ZSt5ctypeIcE _ZNSt5ctypeIcE4_MytE;
#line 2267 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
typedef char _ZNSt5ctypeIcE5_ElemE;
#pragma pack(8)
#line 2261 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt5ctypeIcE { struct _ZSt10ctype_base __b_St10ctype_base;
#line 2468 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
_ZNSt8_Locinfo9_CtypevecE _Ctype;};
#pragma pack()
#pragma pack(8)
#line 797 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocale"
struct _ZSt7codecvtIcciE { struct _ZSt12codecvt_base __b_St12codecvt_base;};
#pragma pack()
#line 133 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
typedef enum _ZNSt7io_errc7io_errcE _ZSt8_Io_errc;
#line 194 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
typedef int _ZNSt10error_code10value_typeE;
#pragma pack(8)
#line 191 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
struct _ZSt10error_code {
#line 282 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
_ZNSt10error_code10value_typeE _Myval;
#line 283 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
const struct _ZSt14error_category *_Mycat;};
#pragma pack()
#pragma pack(8)
#line 502 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
struct _ZSt12system_error { struct _ZSt13runtime_error __b_St13runtime_error;
#line 569 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\system_error"
struct _ZSt10error_code _Mycode;};
#pragma pack()
#pragma pack(8)
#line 51 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZSt5_IosbIiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 206 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
typedef int _ZNSt8ios_base8fmtflagsE;
#line 208 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
typedef int _ZNSt8ios_base8openmodeE;
#pragma pack(8)
#line 222 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZNSt8ios_base7failureE { struct _ZSt12system_error __b_St12system_error;};
#pragma pack()
#pragma pack(8)
#line 202 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZSt8ios_base { const long long *__vptr;
#line 546 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
size_t _Stdstr;
#line 643 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt8ios_base7iostateE _Mystate;
#line 644 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt8ios_base7iostateE _Except;
#line 645 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZNSt8ios_base8fmtflagsE _Fmtfl;
#line 646 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZSt10streamsize _Prec;
#line 647 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
_ZSt10streamsize _Wide;
#line 648 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZNSt8ios_base9_IosarrayE *_Arr;
#line 649 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZNSt8ios_base8_FnarrayE *_Calls;
#line 650 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xiosbase"
struct _ZSt6locale *_Ploc;};
#pragma pack()
#line 83 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
typedef _ZNSt11char_traitsIcE8int_typeE _ZNSt15basic_streambufIcSt11char_traitsIcEE8int_typeE;
#pragma pack(8)
#line 582 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt15basic_streambufIcSt11char_traitsIcEE { const long long *__vptr;
#line 448 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
struct _ZSt6_Mutex _Mylock;
#line 449 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char *_Gfirst;
#line 450 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char *_Pfirst;
#line 451 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char **_IGfirst;
#line 452 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char **_IPfirst;
#line 453 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char *_Gnext;
#line 454 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char *_Pnext;
#line 455 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char **_IGnext;
#line 456 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char **_IPnext;
#line 458 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
int _Gcount;
#line 459 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
int _Pcount;
#line 460 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
int *_IGcount;
#line 461 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
int *_IPcount;
#line 463 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
struct _ZSt6locale *_Plocale;};
#pragma pack()
#pragma pack(8)
#line 624 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE { struct _ZNSt6locale5facetE __b_NSt6locale5facetE;
#line 262 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocnum"
_ZNSt8_Locinfo7_CvtvecE _Cvt;};
#pragma pack()
#pragma pack(8)
#line 627 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE { struct _ZNSt6locale5facetE __b_NSt6locale5facetE;
#line 1070 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xlocnum"
_ZNSt8_Locinfo7_CvtvecE _Cvt;};
#pragma pack()
#line 22 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ios"
typedef struct _ZSo _ZNSt9basic_iosIcSt11char_traitsIcEE5_MyosE;
#line 23 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ios"
typedef struct _ZSt15basic_streambufIcSt11char_traitsIcEE _ZNSt9basic_iosIcSt11char_traitsIcEE5_MysbE;
#line 24 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ios"
typedef struct _ZSt5ctypeIcE _ZNSt9basic_iosIcSt11char_traitsIcEE6_CtypeE;
#pragma pack(8)
#line 573 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt9basic_iosIcSt11char_traitsIcEE { struct _ZSt8ios_base __b_St8ios_base;
#line 172 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ios"
_ZNSt9basic_iosIcSt11char_traitsIcEE5_MysbE *_Mystrbuf;
#line 173 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ios"
_ZNSt9basic_iosIcSt11char_traitsIcEE5_MyosE *_Tiestr;
#line 174 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ios"
char _Fillch;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
typedef struct _ZSo _ZNSo4_MytE;
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
typedef struct _ZSt19ostreambuf_iteratorIcSt11char_traitsIcEE _ZNSo5_IterE;
#line 44 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
typedef struct _ZSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE _ZNSo5_NputE;
#pragma pack(8)
#line 86 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
struct _ZNSo12_Sentry_baseE {
#line 102 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
_ZNSo4_MytE *_Myostr;};
#pragma pack()
#pragma pack(8)
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
struct _ZNSo6sentryE { struct _ZNSo12_Sentry_baseE __b_NSo12_Sentry_baseE;
#line 137 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\ostream"
__nv_bool _Ok;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#pragma pack(8)
#line 588 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSo { const long long *__vptr; struct _ZSt9basic_iosIcSt11char_traitsIcEE __v_St9basic_iosIcSt11char_traitsIcEE;};
#pragma pack()
#pragma pack(8)
#line 588 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct __SO__So { const long long *__vptr;};
#pragma pack()
#line 21 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
typedef struct _ZSi _ZNSi4_MytE;
#line 24 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
typedef struct _ZSt19istreambuf_iteratorIcSt11char_traitsIcEE _ZNSi5_IterE;
#line 25 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
typedef struct _ZSt5ctypeIcE _ZNSi6_CtypeE;
#line 26 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
typedef struct _ZSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE _ZNSi5_NgetE;
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
typedef _ZNSt11char_traitsIcE8int_typeE _ZNSi8int_typeE;
#pragma pack(8)
#line 71 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
struct _ZNSi12_Sentry_baseE {
#line 87 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
_ZNSi4_MytE *_Myistr;};
#pragma pack()
#pragma pack(8)
#line 93 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
struct _ZNSi6sentryE { struct _ZNSi12_Sentry_baseE __b_NSi12_Sentry_baseE;
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
__nv_bool _Ok;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#pragma pack(8)
#line 585 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSi { const long long *__vptr;
#line 859 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
_ZSt10streamsize _Chcount; struct _ZSt9basic_iosIcSt11char_traitsIcEE __v_St9basic_iosIcSt11char_traitsIcEE;};
#pragma pack()
#pragma pack(8)
#line 585 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct __SO__Si { const long long *__vptr;
#line 859 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\istream"
_ZSt10streamsize _Chcount;};
#pragma pack()
#pragma pack(8)
#line 116 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSt15_Allocator_baseIiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 138 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSaIiE10value_typeE *_ZNSaIiE9referenceE;
#line 139 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef const _ZNSaIiE10value_typeE *_ZNSaIiE13const_pointerE;
#line 140 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef const _ZNSaIiE10value_typeE *_ZNSaIiE15const_referenceE;
#line 142 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef size_t _ZNSaIiE9size_typeE;
#line 143 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef ptrdiff_t _ZNSaIiE15difference_typeE;
#line 148 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef struct _ZSaIiE _ZNSaIiE6rebindIiE5otherE;
#pragma pack(8)
#line 130 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSaIiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 421 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE6rebindIiE5otherE _ZNSt11_Vector_valIiSaIiEE5_AltyE;
#line 463 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE15difference_typeE _ZNSt11_Vector_valIiSaIiEE15difference_typeE;
#line 465 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE13const_pointerE _ZNSt11_Vector_valIiSaIiEE13const_pointerE;
#line 467 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE15const_referenceE _ZNSt11_Vector_valIiSaIiEE15const_referenceE;
#line 468 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE10value_typeE _ZNSt11_Vector_valIiSaIiEE10value_typeE;
#pragma pack(8)
#line 417 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt11_Vector_valIiSaIiEE {
#line 470 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIiSaIiEE7pointerE _Myfirst;
#line 471 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIiSaIiEE7pointerE _Mylast;
#line 472 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIiSaIiEE7pointerE _Myend;
#line 473 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIiSaIiEE5_AltyE _Alval;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt6vectorIiSaIiEE _ZNSt6vectorIiSaIiEE4_MytE;
#line 484 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt11_Vector_valIiSaIiEE _ZNSt6vectorIiSaIiEE7_MybaseE;
#line 488 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE9size_typeE _ZNSt6vectorIiSaIiEE9size_typeE;
#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE7pointerE _ZNSt6vectorIiSaIiEE7pointerE;
#line 492 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE9referenceE _ZNSt6vectorIiSaIiEE9referenceE;
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIiE15const_referenceE _ZNSt6vectorIiSaIiEE15const_referenceE;
#line 499 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE _ZNSt6vectorIiSaIiEE8iteratorE;
#line 500 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE _ZNSt6vectorIiSaIiEE14const_iteratorE;
#pragma pack(8)
#line 479 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt6vectorIiSaIiEE { struct _ZSt11_Vector_valIiSaIiEE __b_St11_Vector_valIiSaIiEE;};
#pragma pack()
#pragma pack(8)
#line 116 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSt15_Allocator_baseIfE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 138 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef _ZNSaIfE10value_typeE *_ZNSaIfE9referenceE;
#line 139 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef const _ZNSaIfE10value_typeE *_ZNSaIfE13const_pointerE;
#line 140 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef const _ZNSaIfE10value_typeE *_ZNSaIfE15const_referenceE;
#line 142 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef size_t _ZNSaIfE9size_typeE;
#line 143 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef ptrdiff_t _ZNSaIfE15difference_typeE;
#line 148 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
typedef struct _ZSaIfE _ZNSaIfE6rebindIfE5otherE;
#pragma pack(8)
#line 130 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xmemory"
struct _ZSaIfE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 421 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE6rebindIfE5otherE _ZNSt11_Vector_valIfSaIfEE5_AltyE;
#line 463 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE15difference_typeE _ZNSt11_Vector_valIfSaIfEE15difference_typeE;
#line 465 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE13const_pointerE _ZNSt11_Vector_valIfSaIfEE13const_pointerE;
#line 467 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE15const_referenceE _ZNSt11_Vector_valIfSaIfEE15const_referenceE;
#line 468 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE10value_typeE _ZNSt11_Vector_valIfSaIfEE10value_typeE;
#pragma pack(8)
#line 417 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt11_Vector_valIfSaIfEE {
#line 470 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIfSaIfEE7pointerE _Myfirst;
#line 471 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIfSaIfEE7pointerE _Mylast;
#line 472 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIfSaIfEE7pointerE _Myend;
#line 473 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt11_Vector_valIfSaIfEE5_AltyE _Alval;char __nv_no_debug_dummy_end_padding_0[7];};
#pragma pack()
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt6vectorIfSaIfEE _ZNSt6vectorIfSaIfEE4_MytE;
#line 484 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt11_Vector_valIfSaIfEE _ZNSt6vectorIfSaIfEE7_MybaseE;
#line 488 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE9size_typeE _ZNSt6vectorIfSaIfEE9size_typeE;
#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE7pointerE _ZNSt6vectorIfSaIfEE7pointerE;
#line 492 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE9referenceE _ZNSt6vectorIfSaIfEE9referenceE;
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSaIfE15const_referenceE _ZNSt6vectorIfSaIfEE15const_referenceE;
#line 499 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE _ZNSt6vectorIfSaIfEE8iteratorE;
#line 500 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE _ZNSt6vectorIfSaIfEE14const_iteratorE;
#pragma pack(8)
#line 479 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt6vectorIfSaIfEE { struct _ZSt11_Vector_valIfSaIfEE __b_St11_Vector_valIfSaIfEE;};
#pragma pack()
#line 137 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
typedef struct _ZSt13basic_filebufIcSt11char_traitsIcEE _ZNSt13basic_filebufIcSt11char_traitsIcEE4_MytE;
#line 139 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
typedef _ZNSt11char_traitsIcE10state_typeE _ZNSt13basic_filebufIcSt11char_traitsIcEE5_MystE;
#line 140 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
typedef struct _ZSt7codecvtIcciE _ZNSt13basic_filebufIcSt11char_traitsIcEE4_CvtE;
#line 156 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
typedef _ZNSt11char_traitsIcE8int_typeE _ZNSt13basic_filebufIcSt11char_traitsIcEE8int_typeE;
#pragma pack(8)
#line 610 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt13basic_filebufIcSt11char_traitsIcEE { struct _ZSt15basic_streambufIcSt11char_traitsIcEE __b_St15basic_streambufIcSt11char_traitsIcEE;
#line 655 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
char *_Set_eback;
#line 656 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
char *_Set_egptr;
#line 658 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
const _ZNSt13basic_filebufIcSt11char_traitsIcEE4_CvtE *_Pcvt;
#line 659 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
char _Mychar;
#line 660 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
__nv_bool _Wrotesome;
#line 661 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
_ZNSt11char_traitsIcE10state_typeE _State;
#line 662 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
__nv_bool _Closef;
#line 663 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
struct _iobuf *_Myfile;};
#pragma pack()
#line 928 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
typedef struct _ZSt13basic_filebufIcSt11char_traitsIcEE _ZNSt14basic_ofstreamIcSt11char_traitsIcEE5_MyfbE;
#pragma pack(8)
#line 616 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt14basic_ofstreamIcSt11char_traitsIcEE { struct __SO__So __b_So;
#line 1115 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
_ZNSt14basic_ofstreamIcSt11char_traitsIcEE5_MyfbE _Filebuffer; struct _ZSt9basic_iosIcSt11char_traitsIcEE __v_St9basic_iosIcSt11char_traitsIcEE;};
#pragma pack()
#line 702 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
typedef struct _ZSt13basic_filebufIcSt11char_traitsIcEE _ZNSt14basic_ifstreamIcSt11char_traitsIcEE5_MyfbE;
#pragma pack(8)
#line 613 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt14basic_ifstreamIcSt11char_traitsIcEE { struct __SO__Si __b_Si;
#line 889 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\fstream"
_ZNSt14basic_ifstreamIcSt11char_traitsIcEE5_MyfbE _Filebuffer; struct _ZSt9basic_iosIcSt11char_traitsIcEE __v_St9basic_iosIcSt11char_traitsIcEE;};
#pragma pack()
#line 146 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef float _ZNSt15binary_functionIffbE19first_argument_typeE;
#line 147 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef float _ZNSt15binary_functionIffbE20second_argument_typeE;
#line 148 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef __nv_bool _ZNSt15binary_functionIffbE11result_typeE;
#pragma pack(8)
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIffbE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt7greaterIfE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 134 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt14unary_functionIfbE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 319 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt9binder2ndISt7greaterIfEE {
#line 346 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt7greaterIfE op;
#line 347 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
_ZNSt15binary_functionIffbE20second_argument_typeE value;};
#pragma pack()
#pragma pack(8)
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIiibE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 164 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt10logical_orIiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIfffE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt10multipliesIfE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 352 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt12_Iterator012ISt26random_access_iterator_tagfxPKfRS1_St15_Iterator_base0E {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 29 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE _ZNSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE7_MyiterE;
#line 32 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt11_Vector_valIfSaIfEE7pointerE _ZNSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE5_TptrE;
#pragma pack(8)
#line 20 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE {
#line 256 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE5_TptrE _Ptr;};
#pragma pack()
#line 289 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE _ZNSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE7_MyiterE;
#line 294 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt11_Vector_valIfSaIfEE15difference_typeE _ZNSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE15difference_typeE;
#line 295 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt11_Vector_valIfSaIfEE7pointerE _ZNSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE7pointerE;
#line 307 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE7pointerE _ZNSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE15_Unchecked_typeE;
#pragma pack(8)
#line 285 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt16_Vector_iteratorISt11_Vector_valIfSaIfEEE { struct _ZSt22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE __b_St22_Vector_const_iteratorISt11_Vector_valIfSaIfEEE;};
#pragma pack()
#pragma pack(8)
#line 352 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt12_Iterator012ISt26random_access_iterator_tagixPKiRS1_St15_Iterator_base0E {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 29 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE _ZNSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE7_MyiterE;
#line 32 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt11_Vector_valIiSaIiEE7pointerE _ZNSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE5_TptrE;
#pragma pack(8)
#line 20 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE {
#line 256 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
_ZNSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE5_TptrE _Ptr;};
#pragma pack()
#line 289 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef struct _ZSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE _ZNSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE7_MyiterE;
#line 294 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt11_Vector_valIiSaIiEE15difference_typeE _ZNSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE15difference_typeE;
#line 295 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt11_Vector_valIiSaIiEE7pointerE _ZNSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE7pointerE;
#line 307 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
typedef _ZNSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE7pointerE _ZNSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE15_Unchecked_typeE;
#pragma pack(8)
#line 285 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\vector"
struct _ZSt16_Vector_iteratorISt11_Vector_valIiSaIiEEE { struct _ZSt22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE __b_St22_Vector_const_iteratorISt11_Vector_valIiSaIiEEE;};
#pragma pack()
#line 146 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef int _ZNSt15binary_functionIiiiE19first_argument_typeE;
#line 147 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef int _ZNSt15binary_functionIiiiE20second_argument_typeE;
#line 148 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
typedef int _ZNSt15binary_functionIiiiE11result_typeE;
#pragma pack(8)
#line 144 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt15binary_functionIiiiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt7dividesIiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 134 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstddef"
struct _ZSt14unary_functionIiiE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#pragma pack(8)
#line 319 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt9binder2ndISt7dividesIiEE {
#line 346 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
struct _ZSt7dividesIiE op;
#line 347 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xfunctional"
_ZNSt15binary_functionIiiiE20second_argument_typeE value;};
#pragma pack()
#pragma pack(8)
#line 352 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt12_Iterator012ISt26random_access_iterator_tagcxPKcRS1_St15_Iterator_base0E {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 42 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSaIcE13const_pointerE _ZNSt22_String_const_iteratorIcSt11char_traitsIcESaIcEE7pointerE;
#pragma pack(8)
#line 27 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSt22_String_const_iteratorIcSt11char_traitsIcESaIcEE {
#line 267 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
_ZNSt22_String_const_iteratorIcSt11char_traitsIcESaIcEE7pointerE _Ptr;};
#pragma pack()
#line 316 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
typedef _ZNSs7pointerE _ZNSt16_String_iteratorIcSt11char_traitsIcESaIcEE7pointerE;
#pragma pack(8)
#line 305 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xstring"
struct _ZSt16_String_iteratorIcSt11char_traitsIcESaIcEE { struct _ZSt22_String_const_iteratorIcSt11char_traitsIcESaIcEE __b_St22_String_const_iteratorIcSt11char_traitsIcESaIcEE;};
#pragma pack()
#line 608 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
typedef struct _ZSt15basic_streambufIcSt11char_traitsIcEE _ZNSt19ostreambuf_iteratorIcSt11char_traitsIcEE14streambuf_typeE;
#pragma pack(8)
#line 579 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt19ostreambuf_iteratorIcSt11char_traitsIcEE {
#line 651 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
__nv_bool _Failed;
#line 652 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
_ZNSt19ostreambuf_iteratorIcSt11char_traitsIcEE14streambuf_typeE *_Strbuf;};
#pragma pack()
#pragma pack(8)
#line 336 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
struct _ZSt8iteratorISt18input_iterator_tagcxPcRcE {char __nv_no_debug_dummy_end_padding_0;};
#pragma pack()
#line 485 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
typedef struct _ZSt15basic_streambufIcSt11char_traitsIcEE _ZNSt19istreambuf_iteratorIcSt11char_traitsIcEE14streambuf_typeE;
#pragma pack(8)
#line 576 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\iosfwd"
struct _ZSt19istreambuf_iteratorIcSt11char_traitsIcEE {
#line 567 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
_ZNSt19istreambuf_iteratorIcSt11char_traitsIcEE14streambuf_typeE *_Strbuf;
#line 568 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
__nv_bool _Got;
#line 569 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\streambuf"
char _Val;char __nv_no_debug_dummy_end_padding_0[6];};
#pragma pack()
#line 385 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
typedef _ZNSaIiE10value_typeE _ZNSt15iterator_traitsIPiE10value_typeE;
#line 385 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
typedef _ZNSaIfE10value_typeE _ZNSt15iterator_traitsIPfE10value_typeE;
#line 494 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
typedef struct _ZSt27_Nonscalar_ptr_iterator_tag _ZNSt15_Ptr_cat_helperIbfE5_TypeE;
#line 397 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\xutility"
typedef ptrdiff_t _ZNSt15iterator_traitsIPKiE15difference_typeE;
#line 51 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\new"
extern  __declspec(__device__) void *malloc(size_t);
#line 49 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\new"
extern  __declspec(__device__) void free(void *);

#line 51 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\string.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) void *memcpy(void *, const void *, size_t);
#line 53 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\string.h"

#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\string.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) void *memset(void *, int, size_t);
#line 90 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\string.h"

#line 163 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\time.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) clock_t clock(void);
#line 165 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\time.h"
#line 105 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\malloc.h"
extern  __declspec(__device__) void free(void *);
#line 106 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\malloc.h"
extern  __declspec(__device__) void *malloc(size_t);

#line 238 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) int fprintf(FILE *, const char *, ...);
#line 240 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"

#line 285 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) int printf(const char *, ...);
#line 287 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdio.h"

#line 362 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdlib.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) int abs(int);
#line 364 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdlib.h"

#line 366 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdlib.h"

#line 368 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\stdlib.h"

#line 194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double fabs(double);
#line 196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) int min(int, int);
#line 238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double fmin(double, double);
#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) int max(int, int);
#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double fmax(double, double);
#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double sin(double);
#line 350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double cos(double);
#line 383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double tan(double);
#line 455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double sqrt(double);
#line 524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 540 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1051 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double log(double);
#line 1053 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1055 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1057 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1059 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1061 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1063 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1065 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1069 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1071 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1073 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1075 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1077 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1079 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1687 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double ldexp(double, int);
#line 1689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1691 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1699 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1701 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1703 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 1705 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double frexp(double, int *);
#line 2262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double atan(double);
#line 2776 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2778 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2780 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2782 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2786 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2790 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2792 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2796 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2798 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2800 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2802 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 2804 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3865 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double pow(double, double);
#line 3867 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3921 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double modf(double, double *);
#line 3923 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3925 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3927 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3929 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3933 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3935 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3937 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3939 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3941 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3943 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3945 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3947 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3949 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3951 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3953 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3955 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3957 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3959 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3961 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3963 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3965 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3967 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3969 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3971 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3973 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3975 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3977 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3979 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3981 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3983 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3985 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3987 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3989 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3991 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3993 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3995 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3997 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 3999 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4001 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4003 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4005 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4007 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4009 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4011 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4013 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4015 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4017 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4019 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4021 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4023 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4025 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4027 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4031 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4033 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4035 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4037 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4039 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4041 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4043 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4045 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 4047 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6635 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float atan2f(float, float);
#line 6637 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6639 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6641 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6643 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6645 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6651 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6653 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6645 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float modff(float, float *);
#line 6647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float sqrtf(float);
#line 6649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6651 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6653 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 6655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 161 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double hypot(double, double);
#line 163 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 166 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float hypotf(float, float);
#line 168 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 390 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float frexpf(float, int *);
#line 392 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 394 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float fabsf(float);
#line 396 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 396 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float ldexpf(float, int);
#line 398 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
extern  __declspec(__device__) double _Z8_Pow_intIdET_S0_i(double, int);
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
extern  __declspec(__device__) float _Z8_Pow_intIfET_S0_i(float, int);

#line 486 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 488 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 492 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 494 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 496 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) void __syncthreads(void);
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 233 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 261 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 265 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 277 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 281 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 293 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 297 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 301 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 303 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 305 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 327 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 333 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 337 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 339 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 341 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 345 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 347 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 351 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 417 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 429 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 435 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 437 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 441 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 443 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 451 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 459 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 463 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 465 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 467 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 469 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 471 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 473 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 483 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 485 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 493 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 495 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 499 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 501 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 503 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 505 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 507 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 511 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"

#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) int __iAtomicAdd(int *, int);
#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 72 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 74 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 76 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 78 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 80 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 82 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 86 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 88 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
static  __declspec(__device__) __inline int _ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii(int *, int);

#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float __fAtomicAdd(float *, float);
#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
static  __declspec(__device__) __inline float _ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff(float *, float);

#line 80 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 82 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 86 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 88 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 230 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 248 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 318 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 342 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 446 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 448 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 468 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 474 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 492 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 504 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 510 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 514 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 540 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 554 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 556 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 558 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 564 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 566 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 568 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 572 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 574 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 576 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 578 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 580 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 582 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 586 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 590 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 592 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 594 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 596 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 600 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 602 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 604 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 610 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 614 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 616 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 618 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 622 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 626 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 628 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 630 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 634 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 638 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 642 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 646 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 648 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 650 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 654 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 656 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 660 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 662 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 664 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 668 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 670 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 678 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 680 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 684 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 686 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 704 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 708 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 712 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 714 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 716 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 718 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 720 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 722 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 724 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 726 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 730 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 732 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 734 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 736 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 738 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 740 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 742 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 744 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 746 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 748 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 750 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 752 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 754 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 756 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 758 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 760 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 762 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 764 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 766 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 768 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 770 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 772 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 776 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 778 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 780 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 782 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 786 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 790 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 792 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 796 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 798 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 800 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 802 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 804 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 806 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 808 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 810 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 812 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 814 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 816 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 818 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 820 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 822 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 824 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 826 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 828 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 830 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 832 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 836 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 838 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 840 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 842 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 844 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 848 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 850 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 854 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 856 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 858 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 860 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 862 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 864 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 866 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 868 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 870 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 872 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 874 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 876 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 878 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 880 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 882 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 884 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 886 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 888 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 890 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 892 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 894 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 896 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 898 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 900 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 902 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 904 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 906 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 908 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 910 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 912 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 914 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 916 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 918 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 920 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 922 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 924 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 926 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 928 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 930 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 932 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 934 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 936 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 938 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 940 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 944 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 946 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 948 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 950 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 952 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 954 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 956 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 958 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 960 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 962 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 964 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 966 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 968 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 970 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 972 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 974 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 976 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 978 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 982 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 984 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 986 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 988 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 990 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 992 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 994 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 996 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 998 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1000 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1002 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1004 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1006 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1008 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1010 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1012 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1014 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1016 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1018 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1020 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1022 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1024 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1026 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1028 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1030 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1032 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1034 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1036 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1038 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1040 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1042 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1044 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1046 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1048 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1050 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1052 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1054 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1056 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1058 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1060 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1062 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1064 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1066 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1068 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1070 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1072 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1074 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1076 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1078 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1080 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1082 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1084 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1086 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1088 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1090 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1092 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1094 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1096 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1098 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1230 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1248 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1318 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1342 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1446 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1448 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1468 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1474 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1492 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1504 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1510 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1514 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1540 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1554 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1556 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1558 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1564 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1566 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1568 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1572 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1574 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1576 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1578 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1580 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1582 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1586 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1590 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1592 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1594 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1596 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1600 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1602 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1604 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1610 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1614 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1616 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1618 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1622 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1626 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1628 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1630 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1634 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1638 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1642 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1646 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1648 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1650 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1654 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1656 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1660 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1662 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1664 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1668 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1670 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1678 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1680 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1684 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1686 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1704 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1708 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1712 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1714 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1716 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1718 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1720 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1722 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1724 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1726 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1730 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1732 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1734 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1736 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1738 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1740 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1742 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1744 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1746 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1748 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1750 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1752 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1754 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1756 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1758 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1760 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1762 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1764 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1766 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1768 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1770 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1772 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1776 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1778 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1780 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1782 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1786 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1790 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1792 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1796 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1798 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1800 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1802 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1804 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1806 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1808 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1810 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1812 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1814 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1816 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1818 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1820 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1822 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1824 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1826 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1828 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1830 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1832 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1836 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1838 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1840 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1842 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1844 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1848 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1850 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1854 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1856 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1858 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1860 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1862 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1864 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1866 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1868 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1870 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1872 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1874 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1876 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1878 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1880 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1882 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1884 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1886 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1888 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1890 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1892 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1894 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1896 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1898 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1900 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1902 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1904 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1906 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1908 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1910 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1912 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1914 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1916 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1918 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1920 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1922 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1924 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1926 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1928 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1930 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1932 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1934 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1936 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1938 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1940 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1944 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1946 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1948 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1950 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1952 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1954 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1956 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1958 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1960 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1962 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1964 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1966 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1968 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1970 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1972 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1974 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1976 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1978 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1982 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1984 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1986 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1988 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1990 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1992 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1994 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1996 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1998 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2000 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2002 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2004 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2006 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2008 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2010 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2012 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2014 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2016 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2018 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2020 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2022 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2024 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2026 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2028 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2030 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2032 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2034 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2036 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2038 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2040 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2042 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2044 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2046 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2048 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2050 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2052 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2054 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2056 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2058 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2060 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2062 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2064 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2066 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2068 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2070 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2072 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2074 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2076 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2078 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2080 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2082 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2084 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2086 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2088 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2090 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2092 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2094 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2096 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2098 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2230 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2248 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2318 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2342 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2446 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2448 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2468 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2474 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2492 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2504 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2510 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2514 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2540 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2554 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2556 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2558 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2564 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2566 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2568 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2572 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2574 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2576 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2578 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2580 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2582 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2586 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2590 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2592 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2594 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2596 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2600 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2602 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2604 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2610 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2614 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2616 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2618 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2622 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2626 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2628 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2630 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2634 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2638 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2642 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2646 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2648 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2650 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2654 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2656 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2660 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2662 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2664 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2668 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2670 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2678 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2680 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2684 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2686 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2704 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2708 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2712 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2714 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2716 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2718 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2720 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2722 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2724 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2726 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2730 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2732 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2734 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2736 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2738 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2740 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2742 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2744 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2746 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2748 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2750 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2752 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2754 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2756 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2758 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2760 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2762 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2764 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2766 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2768 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2770 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2772 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2776 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2778 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2780 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2782 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2786 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2790 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2792 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2796 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2798 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2800 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2802 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2804 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2806 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2808 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2810 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2812 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2814 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2816 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2818 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2820 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2822 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2824 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2826 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2828 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2830 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2832 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2836 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2838 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) __nv_bool _Z20calculate_interceptsdddRdS_(double, double, double, double *, double *);
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) void _Z14voxel_walk_GPURPbffffff(__nv_bool **, float, float, float, float, float, float);
#line 2310 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) double _Z15x_remaining_GPUdiRi(double, int, int *);
#line 2317 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) double _Z15y_remaining_GPUdiRi(double, int, int *);
#line 2324 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) double _Z15z_remaining_GPUdiRi(double, int, int *);
#line 4706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) int _Z20position_2_voxel_GPUddd(double, double, double);
#line 4893 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
extern  __declspec(__device__) void _Z16test_func_deviceRiS_S_(int *, int *, int *);
#line 1085 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z30recon_volume_intersections_GPUiPiPbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(int, int *, __nv_bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *);
#line 1407 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z11binning_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(int, int *, int *, __nv_bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *);
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z19calculate_means_GPUPiPfS0_S0_(int *, float *, float *, float *);
#line 1568 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z26sum_squared_deviations_GPUiPiPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_(int, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *);
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z33calculate_standard_deviations_GPUPiPfS0_S0_(int *, float *, float *, float *);
#line 1702 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z20statistical_cuts_GPUiPiS_PfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_Pb(int, int *, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, __nv_bool *);
#line 1766 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z22construct_sinogram_GPUPiPf(int *, float *);
#line 1839 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z10filter_GPUPfS_(float *, float *);
#line 1957 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z18backprojection_GPUPfS_(float *, float *);
#line 2080 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z20FBP_image_2_hull_GPUPfPb(float *, __nv_bool *);
#line 2331 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z17carve_differencesPiS_(int *, int *);
#line 2364 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z6SC_GPUiPbPiS_PfS1_S1_S1_S1_S1_S1_(const int, __nv_bool *, int *, __nv_bool *, float *, float *, float *, float *, float *, float *, float *);
#line 2610 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z7MSC_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_(const int, int *, int *, __nv_bool *, float *, float *, float *, float *, float *, float *, float *);
#line 2793 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z22MSC_edge_detection_GPUPi(int *);
#line 2832 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z6SM_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_(const int, int *, int *, __nv_bool *, float *, float *, float *, float *, float *, float *, float *);
#line 3067 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z21SM_edge_detection_GPUPiS_(int *, int *);
#line 3142 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z23SM_edge_detection_GPU_2PiS_(int *, int *);
#line 4223 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z28create_hull_image_hybrid_GPURPbRPf(__nv_bool **, float **);
#line 4899 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z13test_func_GPUPi(int *);
#line 2108 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z19initialize_hull_GPUIbEvPT_(__nv_bool *);
#line 2108 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z19initialize_hull_GPUIiEvPT_(int *);
#line 3265 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  extern void _Z20averaging_filter_GPUIbEvPT_S1_b(__nv_bool *, __nv_bool *, __nv_bool);
static  __declspec(__device__) const long long _ZTVSt12codecvt_base[7];
static  __declspec(__device__) const long long _ZTVSt10ctype_base[4];
static  __declspec(__device__) const long long _ZTVSt5ctypeIcE[12];
static  __declspec(__device__) const long long _ZTVSt8ios_base[4];
static  __declspec(__device__) const long long _ZTVSt9basic_iosIcSt11char_traitsIcEE[4];
static  __declspec(__device__) const long long _ZTVSo__St14basic_ofstreamIcSt11char_traitsIcEE[5];
static  __declspec(__device__) const long long _ZTVSt9basic_iosIcSt11char_traitsIcEE__So__St14basic_ofstreamIcS1_E[5];
static  __declspec(__device__) const long long _ZTVSi__St14basic_ifstreamIcSt11char_traitsIcEE[5];
static  __declspec(__device__) const long long _ZTVSt9basic_iosIcSt11char_traitsIcEE__Si__St14basic_ifstreamIcS1_E[5];
static  __declspec(__device__) const long long _ZTVSt15basic_streambufIcSt11char_traitsIcEE[18];
static  __declspec(__device__) const long long _ZTVSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE[12];
static  __declspec(__device__) const long long _ZTVSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE[15];
static  __declspec(__device__) const long long *const _ZTTSo[];
static  __declspec(__device__) const long long _ZTVSt7codecvtIcciE[11];
static  __declspec(__device__) const long long *const _ZTTSi[];
static  __declspec(__device__) const long long _ZTVN10__cxxabiv117__class_type_infoE[];
static  __declspec(__device__) const long long _ZTVN10__cxxabiv120__si_class_type_infoE[];
#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\common_functions.h"



























































































































































#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\common_functions.h"








#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\common_functions.h"

#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"






















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 8472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"














































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 14055 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"





#line 14061 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"





#line 14067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"



#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions_dbl_ptx3.h"


























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 4156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions_dbl_ptx3.h"

#line 4158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions_dbl_ptx3.h"

#line 14071 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 14073 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 14075 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\math_functions.h"

#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\common_functions.h"

#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v6.0\\include\\common_functions.h"

#line 3284 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
static  __declspec(__device__) const long long _ZTVSt9bad_alloc[5] = {0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt13runtime_error[5] = {0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVNSt6locale5facetE[4] = {0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt12system_error[5] = {0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVNSt8ios_base7failureE[5] = {0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt14basic_ofstreamIcSt11char_traitsIcEE[10] = {168LL,0LL,0LL,0LL,0LL,(-168LL),(-168LL),0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt14basic_ifstreamIcSt11char_traitsIcEE[10] = {176LL,0LL,0LL,0LL,0LL,(-176LL),(-176LL),0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt13basic_filebufIcSt11char_traitsIcEE[18] = {0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long *const _ZTTSt14basic_ofstreamIcSt11char_traitsIcEE[4] = {(_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE + 3),(_ZTVSo__St14basic_ofstreamIcSt11char_traitsIcEE + 3),(_ZTVSt9basic_iosIcSt11char_traitsIcEE__So__St14basic_ofstreamIcS1_E + 3),(_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE + 8)};
static  __declspec(__device__) const long long *const _ZTTSt14basic_ifstreamIcSt11char_traitsIcEE[4] = {(_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE + 3),(_ZTVSi__St14basic_ifstreamIcSt11char_traitsIcEE + 3),(_ZTVSt9basic_iosIcSt11char_traitsIcEE__Si__St14basic_ifstreamIcS1_E + 3),(_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE + 8)};

#line 3296 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3298 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3300 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3302 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3304 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3306 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3308 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3310 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3312 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3314 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3316 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3318 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3320 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3322 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3324 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3326 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3328 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3330 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3332 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3334 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3336 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3338 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3340 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3342 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3344 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3346 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3348 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3350 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3352 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3354 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3356 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3358 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3360 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3362 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3364 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3366 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3368 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3370 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3372 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3374 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3376 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3378 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3382 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3384 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3386 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3388 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3390 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3392 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3394 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3396 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3398 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3400 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3402 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3404 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3406 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3408 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3412 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3414 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3416 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3418 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3420 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3422 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3424 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3426 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3428 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3430 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3432 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3434 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3436 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3438 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3440 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3442 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3444 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3446 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3448 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3450 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3452 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3454 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3456 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3458 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3460 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3462 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3464 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3466 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3468 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3470 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3472 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3474 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3476 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3478 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3480 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3482 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3484 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3486 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3488 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3490 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3492 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3494 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3496 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3498 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3500 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3502 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3504 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3506 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3508 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3510 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3512 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3518 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3520 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3522 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3524 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3526 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3528 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3530 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3532 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3534 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3536 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3538 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3540 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3542 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3544 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3546 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3548 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3550 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3552 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3554 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3556 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3558 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3560 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3562 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3564 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3566 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3568 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3572 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3574 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3576 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3578 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3580 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3582 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3584 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3586 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3588 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3590 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3592 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3594 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3596 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3598 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3600 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3602 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3604 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3606 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3608 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3610 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3614 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3616 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3618 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3620 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3622 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3624 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3626 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3628 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3630 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3632 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3634 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3636 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3638 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3640 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3642 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3644 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3646 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3648 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3650 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3652 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3654 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3656 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"

#line 3658 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device__) __inline double _Z8_Pow_intIdET_S0_i(
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
double _X, 
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
int _Y){
#line 484 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
{
#line 484 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 unsigned __cuda_local_var_78640_23_non_const__N;
#line 485 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
if (_Y >= 0) {
#line 486 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
__cuda_local_var_78640_23_non_const__N = ((unsigned)_Y); } else  {
#line 488 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
__cuda_local_var_78640_23_non_const__N = ((unsigned)(-_Y)); } {
#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 double _Z;
#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
_Z = (1.0);
#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
for (; ; _X *= _X)
#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
{
#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
if ((__cuda_local_var_78640_23_non_const__N & 1U) != 0U) {
#line 491 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
_Z *= _X; }
#line 492 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
if ((__cuda_local_var_78640_23_non_const__N >>= 1) == 0U) {
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
return (_Y < 0) ? ((1.0) / _Z) : _Z; }
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
} }
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
}}
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 __declspec(__device__) __inline float _Z8_Pow_intIfET_S0_i(
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
float _X, 
#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
int _Y){
#line 484 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
{
#line 484 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 unsigned __cuda_local_var_78640_23_non_const__N;
#line 485 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
if (_Y >= 0) {
#line 486 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
__cuda_local_var_78640_23_non_const__N = ((unsigned)_Y); } else  {
#line 488 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
__cuda_local_var_78640_23_non_const__N = ((unsigned)(-_Y)); } {
#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
 float _Z;
#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
_Z = (1.0F);
#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
for (; ; _X *= _X)
#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
{
#line 490 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
if ((__cuda_local_var_78640_23_non_const__N & 1U) != 0U) {
#line 491 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
_Z *= _X; }
#line 492 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
if ((__cuda_local_var_78640_23_non_const__N >>= 1) == 0U) {
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
return (_Y < 0) ? ((1.0F) / _Z) : _Z; }
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
} }
#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
}}

#line 496 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 498 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 500 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 502 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 504 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 506 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 508 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 510 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 512 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 514 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 516 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 518 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 520 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 522 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 524 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 526 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 528 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 530 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 532 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 534 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 536 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 538 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 540 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 542 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 544 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 546 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 548 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 550 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 552 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 554 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 556 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 558 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 560 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 562 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 564 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 566 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 568 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 570 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 572 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 574 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 576 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 578 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 580 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 582 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 584 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 586 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 588 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 590 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 592 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 594 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 596 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 598 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 600 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 602 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 604 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 606 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 608 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 610 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 612 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 614 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 616 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 618 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 620 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 622 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 624 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 626 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 628 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 630 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 632 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 634 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 636 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 638 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 640 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 642 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 644 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 646 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 648 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 650 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 652 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 654 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 656 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 658 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 660 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 662 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 664 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 666 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 668 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 670 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 672 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 674 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 676 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 678 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 680 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 682 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 684 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 686 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 688 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 690 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 692 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 694 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 696 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 698 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 700 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 702 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 704 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 706 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 708 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 710 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 712 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 714 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 716 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 718 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 720 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 722 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 724 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 726 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 728 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 730 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 732 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 734 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 736 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 738 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 740 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 742 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 744 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 746 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 748 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 750 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 752 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 754 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 756 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 758 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 760 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 762 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 764 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 766 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 768 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 770 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 772 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 774 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 776 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 778 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 780 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 782 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 784 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 786 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 788 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 790 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 792 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 794 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 796 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 798 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 800 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 802 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 804 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 806 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 808 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 810 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 812 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 814 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 816 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 818 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 820 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 822 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 824 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 826 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 828 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 830 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 832 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 834 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 836 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 838 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 840 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 842 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 844 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 846 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 848 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 850 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 852 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 854 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 856 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 858 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 860 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 862 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 864 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 866 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 868 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 870 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 872 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 874 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 876 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 878 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 880 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 882 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 884 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 886 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 888 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 890 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 892 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 894 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 896 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 898 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 900 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 902 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 904 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 906 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 908 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 910 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 912 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 914 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 916 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 918 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 920 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 922 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"

#line 924 "C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\\math.h"
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
static  __declspec(__device__) __inline int _ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii(
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
int *address, 
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
int val){
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
{
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
return __iAtomicAdd(address, val);
#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
}}

#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"

#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_11_atomic_functions.h"
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
static  __declspec(__device__) __inline float _ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff(
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
float *address, 
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
float val){
#line 78 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
{
#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
return __fAtomicAdd(address, val);
#line 80 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
}}

#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 85 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 89 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 233 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 261 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 265 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 277 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 281 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 293 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 297 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 301 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 303 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 305 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 327 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 333 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 337 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 339 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 341 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 345 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 347 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 351 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 417 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 429 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 435 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 437 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 441 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 443 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 451 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 459 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 463 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 465 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 467 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 469 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 471 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 473 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 483 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 485 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 493 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 495 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 499 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 501 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 503 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 505 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 507 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 511 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 513 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 515 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 517 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 519 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 521 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 525 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 531 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 533 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 535 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 537 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 539 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 541 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 543 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 545 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 547 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 549 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 551 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 553 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 555 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 557 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 559 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 561 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 563 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 565 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 567 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 569 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 571 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 573 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 575 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 577 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 579 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 581 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 583 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 585 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 587 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 589 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 591 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 593 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 595 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 597 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 599 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 601 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 603 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 605 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 607 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 609 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 611 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 613 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 615 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 617 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 619 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 621 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 623 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 625 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 627 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 629 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 633 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 635 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 637 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 639 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 641 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 643 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 645 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 651 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 653 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 657 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 659 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 661 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 663 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 665 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 667 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 669 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 671 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 673 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 677 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 683 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 685 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 687 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 691 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 699 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 701 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 703 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 705 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 707 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 709 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 711 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 713 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 715 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 717 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 719 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 721 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 723 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 725 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 727 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 729 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 731 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 733 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 735 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 737 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 739 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 741 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 745 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 747 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 749 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 751 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 753 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 755 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 757 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 759 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 761 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 763 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 765 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 767 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 769 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 771 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 773 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 775 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 777 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 779 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 781 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 783 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 785 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 787 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 789 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 791 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 795 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 797 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 799 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 801 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 803 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 805 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 807 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 809 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 811 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 813 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 815 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 817 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 819 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 821 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 823 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 825 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 827 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 829 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 833 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 835 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 837 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 839 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 841 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 843 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 845 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 847 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 849 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 851 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 853 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 855 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 857 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 859 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 861 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 863 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 865 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 867 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 869 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 871 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 873 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 875 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 877 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 879 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 881 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 883 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 885 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 887 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 889 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 891 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 893 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 895 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 897 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 899 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 901 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 903 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 905 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 907 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 909 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 911 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 913 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 915 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 917 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 919 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 921 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 923 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 925 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 927 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 929 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 933 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 935 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 937 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 939 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 941 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 943 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 945 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 947 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 949 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 951 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 953 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 955 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 957 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 959 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 961 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 963 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 965 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 967 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 969 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 971 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 973 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 975 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 977 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 979 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 981 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 983 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 985 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 987 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 989 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 991 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 993 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 995 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 997 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 999 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1001 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1003 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1005 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1007 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1009 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1011 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1013 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1015 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1017 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1019 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1021 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1023 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1025 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1027 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1031 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1033 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1035 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1037 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1039 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1041 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1043 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1045 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1047 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1049 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1051 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1053 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1055 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1057 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1059 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1061 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1063 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1065 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1069 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1071 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1073 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1075 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1077 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1079 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1081 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1083 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1085 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1087 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1089 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1091 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1093 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1095 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1097 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1099 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1233 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1261 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1265 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1277 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1281 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1293 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1297 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1301 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1303 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1305 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1327 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1333 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1337 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1339 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1341 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1345 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1347 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1351 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1417 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1429 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1435 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1437 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1441 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1443 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1451 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1459 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1463 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1465 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1467 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1469 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1471 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1473 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1483 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1485 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1493 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1495 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1499 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1501 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1503 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1505 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1507 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1511 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1513 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1515 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1517 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1519 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1521 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1525 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1531 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1533 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1535 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1537 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1539 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1541 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1543 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1545 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1547 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1549 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1551 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1553 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1555 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1557 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1559 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1561 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1563 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1565 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1567 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1569 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1571 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1573 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1575 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1577 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1579 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1581 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1583 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1585 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1587 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1589 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1591 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1593 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1595 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1597 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1599 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1601 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1603 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1605 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1607 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1609 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1611 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1613 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1615 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1617 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1619 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1621 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1623 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1625 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1627 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1629 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1633 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1635 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1637 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1639 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1641 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1643 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1645 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1651 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1653 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1657 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1659 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1661 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1663 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1665 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1667 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1669 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1671 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1673 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1677 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1683 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1685 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1687 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1691 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1699 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1701 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1703 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1705 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1707 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1709 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1711 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1713 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1715 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1717 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1719 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1721 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1723 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1725 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1727 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1729 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1731 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1733 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1735 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1737 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1739 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1741 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1745 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1747 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1749 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1751 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1753 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1755 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1757 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1759 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1761 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1763 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1765 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1767 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1769 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1771 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1773 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1775 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1777 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1779 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1781 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1783 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1785 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1787 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1789 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1791 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1795 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1797 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1799 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1801 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1803 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1805 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1807 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1809 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1811 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1813 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1815 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1817 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1819 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1821 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1823 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1825 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1827 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1829 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1833 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1835 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1837 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1839 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1841 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1843 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1845 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1847 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1849 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1851 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1853 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1855 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1857 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1859 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1861 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1863 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1865 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1867 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1869 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1871 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1873 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1875 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1877 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1879 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1881 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1883 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1885 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1887 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1889 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1891 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1893 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1895 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1897 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1899 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1901 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1903 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1905 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1907 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1909 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1911 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1913 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1915 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1917 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1919 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1921 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1923 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1925 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1927 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1929 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1933 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1935 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1937 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1939 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1941 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1943 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1945 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1947 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1949 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1951 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1953 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1955 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1957 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1959 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1961 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1963 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1965 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1967 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1969 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1971 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1973 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1975 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1977 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1979 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1981 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1983 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1985 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1987 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1989 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1991 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1993 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1995 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1997 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 1999 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2001 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2003 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2005 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2007 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2009 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2011 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2013 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2015 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2017 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2019 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2021 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2023 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2025 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2027 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2031 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2033 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2035 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2037 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2039 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2041 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2043 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2045 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2047 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2049 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2051 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2053 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2055 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2057 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2059 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2061 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2063 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2065 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2069 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2071 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2073 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2075 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2077 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2079 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2081 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2083 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2085 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2087 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2089 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2091 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2093 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2095 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2097 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2099 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2233 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2261 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2265 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2277 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2281 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2293 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2297 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2301 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2303 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2305 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2327 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2333 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2337 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2339 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2341 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2345 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2347 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2351 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2417 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2429 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2435 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2437 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2441 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2443 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2451 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2459 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2463 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2465 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2467 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2469 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2471 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2473 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2483 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2485 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2493 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2495 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2499 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2501 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2503 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2505 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2507 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2511 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2513 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2515 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2517 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2519 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2521 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2525 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2531 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2533 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2535 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2537 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2539 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2541 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2543 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2545 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2547 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2549 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2551 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2553 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2555 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2557 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2559 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2561 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2563 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2565 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2567 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2569 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2571 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2573 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2575 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2577 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2579 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2581 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2583 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2585 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2587 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2589 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2591 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2593 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2595 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2597 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2599 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2601 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2603 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2605 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2607 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2609 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2611 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2613 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2615 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2617 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2619 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2621 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2623 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2625 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2627 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2629 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2633 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2635 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2637 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2639 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2641 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2643 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2645 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2651 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2653 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2657 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2659 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2661 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2663 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2665 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2667 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2669 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2671 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2673 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2677 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2683 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2685 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2687 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2691 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2699 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2701 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2703 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2705 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2707 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2709 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2711 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2713 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2715 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2717 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2719 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2721 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2723 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2725 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2727 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2729 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2731 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2733 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2735 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2737 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2739 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2741 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2745 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2747 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2749 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2751 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2753 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2755 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2757 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2759 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2761 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2763 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2765 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2767 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2769 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2771 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2773 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2775 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2777 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2779 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2781 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2783 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2785 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2787 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2789 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2791 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2795 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2797 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2799 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2801 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2803 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2805 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2807 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2809 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2811 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2813 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2815 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2817 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2819 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2821 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2823 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2825 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2827 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2829 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2833 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2835 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2837 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2839 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"

#line 2841 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\sm_20_atomic_functions.h"
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) __nv_bool _Z20calculate_interceptsdddRdS_(
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double u, 
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double t, 
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double ut_angle, 
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double *u_intercept, 
#line 1209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double *t_intercept){
#line 1210 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1241 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_366691_7_non_const_entry_in_cone;
#line 1245 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366695_9_non_const_u_temp;
#line 1253 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366703_9_non_const_m;
#line 1254 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366704_9_non_const_b_in;
#line 1257 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366707_9_non_const_a;
#line 1258 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366708_9_non_const_b;
#line 1259 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366709_9_non_const_c;
#line 1260 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366710_9_non_const_entry_discriminant;
#line 1261 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_366711_7_non_const_intersected;
#line 1241 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366691_7_non_const_entry_in_cone = ((__nv_bool)(((ut_angle > (atan((1.0)))) && (ut_angle < ((3.0) * (atan((1.0)))))) || ((ut_angle > ((5.0) * (atan((1.0))))) && (ut_angle < ((7.0) * (atan((1.0))))))));
#line 1246 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366691_7_non_const_entry_in_cone)
#line 1247 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1248 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366695_9_non_const_u_temp = u;
#line 1249 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
u = (-t);
#line 1250 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
t = __cuda_local_var_366695_9_non_const_u_temp;
#line 1251 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
ut_angle += ((2.0) * (atan((1.0))));
#line 1252 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1253 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366703_9_non_const_m = (tan(ut_angle));
#line 1254 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366704_9_non_const_b_in = (t - (__cuda_local_var_366703_9_non_const_m * u));
#line 1257 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366707_9_non_const_a = ((1.0) + (_Z8_Pow_intIdET_S0_i(__cuda_local_var_366703_9_non_const_m, 2)));
#line 1258 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366708_9_non_const_b = (((2.0) * __cuda_local_var_366703_9_non_const_m) * __cuda_local_var_366704_9_non_const_b_in);
#line 1259 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366709_9_non_const_c = ((_Z8_Pow_intIdET_S0_i(__cuda_local_var_366704_9_non_const_b_in, 2)) - (_Z8_Pow_intIdET_S0_i((8.0), 2)));
#line 1260 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366710_9_non_const_entry_discriminant = ((_Z8_Pow_intIdET_S0_i(__cuda_local_var_366708_9_non_const_b, 2)) - (((4.0) * __cuda_local_var_366707_9_non_const_a) * __cuda_local_var_366709_9_non_const_c));
#line 1261 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366711_7_non_const_intersected = ((__nv_bool)(__cuda_local_var_366710_9_non_const_entry_discriminant > (0.0)));
#line 1274 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366711_7_non_const_intersected)
#line 1275 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T228;
 double __T229;
 double __T230;
 double __T231;
#line 1276 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366726_10_non_const_u_intercept_1;
#line 1277 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366727_10_non_const_u_intercept_2;
#line 1278 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366728_10_non_const_t_intercept_1;
#line 1279 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366729_10_non_const_t_intercept_2;
#line 1280 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366730_10_non_const_squared_distance_1;
#line 1281 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366731_10_non_const_squared_distance_2;
#line 1276 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366726_10_non_const_u_intercept_1 = ( fdivide(((sqrt(__cuda_local_var_366710_9_non_const_entry_discriminant)) - __cuda_local_var_366708_9_non_const_b) , ((2.0) * __cuda_local_var_366707_9_non_const_a)));
#line 1277 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366727_10_non_const_u_intercept_2 = ( fdivide(((sqrt(__cuda_local_var_366710_9_non_const_entry_discriminant)) + __cuda_local_var_366708_9_non_const_b) , ((2.0) * __cuda_local_var_366707_9_non_const_a)));
#line 1278 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366728_10_non_const_t_intercept_1 = ((__cuda_local_var_366703_9_non_const_m * __cuda_local_var_366726_10_non_const_u_intercept_1) + __cuda_local_var_366704_9_non_const_b_in);
#line 1279 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366729_10_non_const_t_intercept_2 = ((__cuda_local_var_366703_9_non_const_m * __cuda_local_var_366727_10_non_const_u_intercept_2) - __cuda_local_var_366704_9_non_const_b_in);
#line 1280 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366730_10_non_const_squared_distance_1 = (((__T228 = (__cuda_local_var_366726_10_non_const_u_intercept_1 - u)) , (_Z8_Pow_intIdET_S0_i(__T228, 2))) + ((__T229 = (__cuda_local_var_366728_10_non_const_t_intercept_1 - t)) , (_Z8_Pow_intIdET_S0_i(__T229, 2))));
#line 1281 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366731_10_non_const_squared_distance_2 = (((__T230 = (__cuda_local_var_366727_10_non_const_u_intercept_2 + u)) , (_Z8_Pow_intIdET_S0_i(__T230, 2))) + ((__T231 = (__cuda_local_var_366729_10_non_const_t_intercept_2 + t)) , (_Z8_Pow_intIdET_S0_i(__T231, 2))));
#line 1282 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*u_intercept) = ((__cuda_local_var_366726_10_non_const_u_intercept_1 * ((double)(__cuda_local_var_366730_10_non_const_squared_distance_1 <= __cuda_local_var_366731_10_non_const_squared_distance_2))) - (__cuda_local_var_366727_10_non_const_u_intercept_2 * ((double)(__cuda_local_var_366730_10_non_const_squared_distance_1 > __cuda_local_var_366731_10_non_const_squared_distance_2))));
#line 1283 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*t_intercept) = ((__cuda_local_var_366728_10_non_const_t_intercept_1 * ((double)(__cuda_local_var_366730_10_non_const_squared_distance_1 <= __cuda_local_var_366731_10_non_const_squared_distance_2))) - (__cuda_local_var_366729_10_non_const_t_intercept_2 * ((double)(__cuda_local_var_366730_10_non_const_squared_distance_1 > __cuda_local_var_366731_10_non_const_squared_distance_2))));
#line 1284 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1286 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366691_7_non_const_entry_in_cone)
#line 1287 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1288 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366695_9_non_const_u_temp = (*u_intercept);
#line 1289 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*u_intercept) = (*t_intercept);
#line 1290 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*t_intercept) = (-__cuda_local_var_366695_9_non_const_u_temp);
#line 1291 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
ut_angle -= ((2.0) * (atan((1.0))));
#line 1292 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1294 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
return __cuda_local_var_366711_7_non_const_intersected;
#line 1295 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) void _Z14voxel_walk_GPURPbffffff(
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool **image, 
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float x_entry, 
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float y_entry, 
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float z_entry, 
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float x_exit, 
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float y_exit, 
#line 2140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float z_exit){
#line 2141 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T260;
 float __T261;
 float __T262;
#line 2145 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367595_6_non_const_x_move_direction;
#line 2145 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367595_24_non_const_y_move_direction;
#line 2145 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367595_42_non_const_z_move_direction;
#line 2146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367596_8_non_const_delta_yx;
#line 2146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367596_18_non_const_delta_zx;
#line 2146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367596_28_non_const_delta_zy;
#line 2151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367601_8_non_const_x;
#line 2151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367601_21_non_const_y;
#line 2151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367601_34_non_const_z;
#line 2152 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367602_8_non_const_x_to_go;
#line 2152 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367602_17_non_const_y_to_go;
#line 2152 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367602_26_non_const_z_to_go;
#line 2153 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367603_8_non_const_x_extension;
#line 2153 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367603_21_non_const_y_extension;
#line 2154 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367604_6_non_const_voxel_x;
#line 2154 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367604_15_non_const_voxel_y;
#line 2154 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367604_24_non_const_voxel_z;
#line 2155 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367605_6_non_const_voxel;
#line 2155 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367605_13_non_const_voxel_x_out;
#line 2155 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367605_26_non_const_voxel_y_out;
#line 2155 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367605_39_non_const_voxel_z_out;
#line 2155 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367605_52_non_const_voxel_out;
#line 2156 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367606_7_non_const_end_walk;
#line 2151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367601_8_non_const_x = x_entry;
#line 2151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367601_21_non_const_y = y_entry;
#line 2151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367601_34_non_const_z = z_entry;
#line 2160 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367595_6_non_const_x_move_direction = (((int)(x_entry <= x_exit)) - ((int)(x_entry > x_exit)));
#line 2161 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367595_24_non_const_y_move_direction = (((int)(y_entry <= y_exit)) - ((int)(y_entry > y_exit)));
#line 2162 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367595_42_non_const_z_move_direction = (((int)(z_entry <= z_exit)) - ((int)(z_entry > z_exit)));
#line 2164 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go = ((float)(_Z15x_remaining_GPUdiRi(((double)__cuda_local_var_367601_8_non_const_x), __cuda_local_var_367595_6_non_const_x_move_direction, (&__cuda_local_var_367604_6_non_const_voxel_x))));
#line 2165 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go = ((float)(_Z15y_remaining_GPUdiRi(((double)__cuda_local_var_367601_21_non_const_y), (-__cuda_local_var_367595_24_non_const_y_move_direction), (&__cuda_local_var_367604_15_non_const_voxel_y))));
#line 2166 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_26_non_const_z_to_go = ((float)(_Z15z_remaining_GPUdiRi(((double)__cuda_local_var_367601_34_non_const_z), (-__cuda_local_var_367595_42_non_const_z_move_direction), (&__cuda_local_var_367604_24_non_const_voxel_z))));
#line 2167 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_6_non_const_voxel = ((__cuda_local_var_367604_6_non_const_voxel_x + (__cuda_local_var_367604_15_non_const_voxel_y * 200)) + ((__cuda_local_var_367604_24_non_const_voxel_z * 200) * 200));
#line 2172 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367596_8_non_const_delta_yx = ((__T260 = ( fdividef((y_exit - y_entry) , (x_exit - x_entry)))) , (fabsf(__T260)));
#line 2173 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367596_18_non_const_delta_zx = ((__T261 = ( fdividef((z_exit - z_entry) , (x_exit - x_entry)))) , (fabsf(__T261)));
#line 2174 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367596_28_non_const_delta_zy = ((__T262 = ( fdividef((z_exit - z_entry) , (y_exit - y_entry)))) , (fabsf(__T262)));
#line 2178 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_13_non_const_voxel_x_out = ((int)__double2int_rz((double)(( fdivide((((double)x_exit) + (8.0)) , (0.080000000000000002))))));
#line 2179 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_26_non_const_voxel_y_out = ((int)__double2int_rz((double)(( fdivide(((8.0) - ((double)y_exit)) , (0.080000000000000002))))));
#line 2180 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_39_non_const_voxel_z_out = ((int)__double2int_rz((double)(( fdivide(((3.0) - ((double)z_exit)) , (0.25))))));
#line 2181 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_52_non_const_voxel_out = ((int)((__cuda_local_var_367605_13_non_const_voxel_x_out + (__cuda_local_var_367605_26_non_const_voxel_y_out * 200)) + ((__cuda_local_var_367605_39_non_const_voxel_z_out * 200) * 200)));
#line 2183 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367606_7_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_367605_6_non_const_voxel == __cuda_local_var_367605_52_non_const_voxel_out) || (__cuda_local_var_367604_6_non_const_voxel_x >= 200)) || (__cuda_local_var_367604_15_non_const_voxel_y >= 200)) || (__cuda_local_var_367604_24_non_const_voxel_z >= 24)));
#line 2184 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_367606_7_non_const_end_walk)) {
#line 2185 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
((*image)[__cuda_local_var_367605_6_non_const_voxel]) = ((__nv_bool)0); }
#line 2189 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (z_entry != z_exit)
#line 2190 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2191 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_367606_7_non_const_end_walk))
#line 2192 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2194 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367603_8_non_const_x_extension = (__cuda_local_var_367596_18_non_const_delta_zx * __cuda_local_var_367602_8_non_const_x_to_go);
#line 2195 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367603_21_non_const_y_extension = (__cuda_local_var_367596_28_non_const_delta_zy * __cuda_local_var_367602_17_non_const_y_to_go);
#line 2196 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_367602_26_non_const_z_to_go <= __cuda_local_var_367603_8_non_const_x_extension) && (__cuda_local_var_367602_26_non_const_z_to_go <= __cuda_local_var_367603_21_non_const_y_extension))
#line 2197 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2204 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go -= ( fdividef(__cuda_local_var_367602_26_non_const_z_to_go , __cuda_local_var_367596_18_non_const_delta_zx));
#line 2205 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go -= ( fdividef(__cuda_local_var_367602_26_non_const_z_to_go , __cuda_local_var_367596_28_non_const_delta_zy));
#line 2206 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_26_non_const_z_to_go = (0.25F);
#line 2207 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_24_non_const_voxel_z -= __cuda_local_var_367595_42_non_const_z_move_direction;
#line 2208 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367602_8_non_const_x_to_go <= (0.0F))
#line 2209 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2210 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_6_non_const_voxel_x += __cuda_local_var_367595_6_non_const_x_move_direction;
#line 2211 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go = (0.07999999821F);
#line 2212 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2213 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367602_17_non_const_y_to_go <= (0.0F))
#line 2214 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2215 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_15_non_const_voxel_y -= __cuda_local_var_367595_24_non_const_y_move_direction;
#line 2216 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go = (0.07999999821F);
#line 2217 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2218 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} else  {
#line 2220 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367603_8_non_const_x_extension <= __cuda_local_var_367603_21_non_const_y_extension)
#line 2221 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2226 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go = (0.07999999821F);
#line 2227 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go -= (__cuda_local_var_367596_8_non_const_delta_yx * __cuda_local_var_367602_8_non_const_x_to_go);
#line 2228 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_26_non_const_z_to_go -= (__cuda_local_var_367596_18_non_const_delta_zx * __cuda_local_var_367602_8_non_const_x_to_go);
#line 2232 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_6_non_const_voxel_x += __cuda_local_var_367595_6_non_const_x_move_direction;
#line 2233 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367602_17_non_const_y_to_go <= (0.0F))
#line 2234 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2235 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go = (0.07999999821F);
#line 2236 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_15_non_const_voxel_y -= __cuda_local_var_367595_24_non_const_y_move_direction;
#line 2237 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2238 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2241 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2241 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2246 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go -= ( fdividef(__cuda_local_var_367602_17_non_const_y_to_go , __cuda_local_var_367596_8_non_const_delta_yx));
#line 2247 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go = (0.07999999821F);
#line 2248 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_26_non_const_z_to_go -= (__cuda_local_var_367596_28_non_const_delta_zy * __cuda_local_var_367602_17_non_const_y_to_go);
#line 2252 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_15_non_const_voxel_y -= __cuda_local_var_367595_24_non_const_y_move_direction;
#line 2253 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2257 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_6_non_const_voxel = ((__cuda_local_var_367604_6_non_const_voxel_x + (__cuda_local_var_367604_15_non_const_voxel_y * 200)) + ((__cuda_local_var_367604_24_non_const_voxel_z * 200) * 200));
#line 2258 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367606_7_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_367605_6_non_const_voxel == __cuda_local_var_367605_52_non_const_voxel_out) || (__cuda_local_var_367604_6_non_const_voxel_x >= 200)) || (__cuda_local_var_367604_15_non_const_voxel_y >= 200)) || (__cuda_local_var_367604_24_non_const_voxel_z >= 24)));
#line 2259 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_367606_7_non_const_end_walk)) {
#line 2260 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
((*image)[__cuda_local_var_367605_6_non_const_voxel]) = ((__nv_bool)0); }
#line 2261 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2262 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2264 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2264 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2268 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_367606_7_non_const_end_walk))
#line 2269 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2271 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367603_21_non_const_y_extension = ( fdividef(__cuda_local_var_367602_17_non_const_y_to_go , __cuda_local_var_367596_8_non_const_delta_yx));
#line 2273 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367602_8_non_const_x_to_go <= __cuda_local_var_367603_21_non_const_y_extension)
#line 2274 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2278 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go = (0.07999999821F);
#line 2279 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go -= (__cuda_local_var_367596_8_non_const_delta_yx * __cuda_local_var_367602_8_non_const_x_to_go);
#line 2281 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_6_non_const_voxel_x += __cuda_local_var_367595_6_non_const_x_move_direction;
#line 2282 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367602_17_non_const_y_to_go <= (0.0F))
#line 2283 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2284 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go = (0.07999999821F);
#line 2285 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_15_non_const_voxel_y -= __cuda_local_var_367595_24_non_const_y_move_direction;
#line 2286 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2287 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2290 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2290 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2295 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_8_non_const_x_to_go -= ( fdividef(__cuda_local_var_367602_17_non_const_y_to_go , __cuda_local_var_367596_8_non_const_delta_yx));
#line 2296 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367602_17_non_const_y_to_go = (0.07999999821F);
#line 2297 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367604_15_non_const_voxel_y -= __cuda_local_var_367595_24_non_const_y_move_direction;
#line 2298 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2301 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367605_6_non_const_voxel = ((__cuda_local_var_367604_6_non_const_voxel_x + (__cuda_local_var_367604_15_non_const_voxel_y * 200)) + ((__cuda_local_var_367604_24_non_const_voxel_z * 200) * 200));
#line 2302 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367606_7_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_367605_6_non_const_voxel == __cuda_local_var_367605_52_non_const_voxel_out) || (__cuda_local_var_367604_6_non_const_voxel_x >= 200)) || (__cuda_local_var_367604_15_non_const_voxel_y >= 200)) || (__cuda_local_var_367604_24_non_const_voxel_z >= 24)));
#line 2303 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_367606_7_non_const_end_walk)) {
#line 2304 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
((*image)[__cuda_local_var_367605_6_non_const_voxel]) = ((__nv_bool)0); }
#line 2306 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2308 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 2309 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2310 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) double _Z15x_remaining_GPUdiRi(
#line 2310 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double x, 
#line 2310 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int x_move_direction, 
#line 2310 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *voxel_x){
#line 2311 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2312 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367762_9_non_const_voxel_x_float;
#line 2313 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367763_9_non_const_x_inside;
#line 2313 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367763_9_non_const_x_inside = ((modf(( fdivide((x + (8.0)) , (0.080000000000000002))), (&__cuda_local_var_367762_9_non_const_voxel_x_float))) * (0.080000000000000002));
#line 2314 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*voxel_x) = ((int)__double2int_rz((double)(__cuda_local_var_367762_9_non_const_voxel_x_float)));
#line 2315 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
return (((double)(x_move_direction > 0)) * ((0.080000000000000002) - __cuda_local_var_367763_9_non_const_x_inside)) + (((double)(x_move_direction <= 0)) * __cuda_local_var_367763_9_non_const_x_inside);
#line 2316 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2317 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) double _Z15y_remaining_GPUdiRi(
#line 2317 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double y, 
#line 2317 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int y_move_direction, 
#line 2317 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *voxel_y){
#line 2318 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2319 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367769_9_non_const_voxel_y_float;
#line 2320 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367770_9_non_const_y_inside;
#line 2320 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367770_9_non_const_y_inside = ((modf(( fdivide(((8.0) - y) , (0.080000000000000002))), (&__cuda_local_var_367769_9_non_const_voxel_y_float))) * (0.080000000000000002));
#line 2321 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*voxel_y) = ((int)__double2int_rz((double)(__cuda_local_var_367769_9_non_const_voxel_y_float)));
#line 2322 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
return (((double)(y_move_direction > 0)) * ((0.080000000000000002) - __cuda_local_var_367770_9_non_const_y_inside)) + (((double)(y_move_direction <= 0)) * __cuda_local_var_367770_9_non_const_y_inside);
#line 2323 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2324 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) double _Z15z_remaining_GPUdiRi(
#line 2324 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double z, 
#line 2324 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int z_move_direction, 
#line 2324 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *voxel_z){
#line 2325 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T263;
#line 2326 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367776_8_non_const_voxel_z_float;
#line 2327 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_367777_8_non_const_z_inside;
#line 2327 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367777_8_non_const_z_inside = ((float)(((double)((__T263 = ((float)( fdivide(((3.0) - z) , (0.25))))) , (modff(__T263, (&__cuda_local_var_367776_8_non_const_voxel_z_float))))) * (0.25)));
#line 2328 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*voxel_z) = ((int)__float2int_rz((float)(__cuda_local_var_367776_8_non_const_voxel_z_float)));
#line 2329 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
return (((double)(z_move_direction > 0)) * ((0.25) - ((double)__cuda_local_var_367777_8_non_const_z_inside))) + ((double)(((float)(z_move_direction <= 0)) * __cuda_local_var_367777_8_non_const_z_inside));
#line 2330 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 4706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) int _Z20position_2_voxel_GPUddd(
#line 4706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double x, 
#line 4706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double y, 
#line 4706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
double z){
#line 4707 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 4708 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_370158_6_non_const_voxel_x;
#line 4709 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_370159_6_non_const_voxel_y;
#line 4710 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_370160_6_non_const_voxel_z;
#line 4708 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370158_6_non_const_voxel_x = ((int)__double2int_rz((double)(( fdivide((x + (8.0)) , (0.080000000000000002))))));
#line 4709 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370159_6_non_const_voxel_y = ((int)__double2int_rz((double)(( fdivide(((8.0) - y) , (0.080000000000000002))))));
#line 4710 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370160_6_non_const_voxel_z = ((int)__double2int_rz((double)(( fdivide(((3.0) - z) , (0.25))))));
#line 4711 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
return (__cuda_local_var_370158_6_non_const_voxel_x + (__cuda_local_var_370159_6_non_const_voxel_y * 200)) + ((__cuda_local_var_370160_6_non_const_voxel_z * 200) * 200);
#line 4712 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 4893 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __declspec(__device__) void _Z16test_func_deviceRiS_S_(
#line 4893 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *x, 
#line 4893 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *y, 
#line 4893 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *z){
#line 4894 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 4895 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*x) = 2;
#line 4896 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*y) = 3;
#line 4897 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(*z) = 4; 
#line 4898 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1085 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z30recon_volume_intersections_GPUiPiPbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int num_histories, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *gantry_angle, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *missed_recon_volume, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *t_in_1, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *t_in_2, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *t_out_1, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *t_out_2, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *u_in_1, 
#line 1087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *u_in_2, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *u_out_1, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *u_out_2, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *v_in_1, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *v_in_2, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *v_out_1, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *v_out_2, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_entry, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_entry, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_entry, 
#line 1088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_exit, 
#line 1089 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_exit, 
#line 1089 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_exit, 
#line 1089 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_entry_angle, 
#line 1089 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_entry_angle, 
#line 1089 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_exit_angle, 
#line 1089 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_exit_angle){
#line 1091 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1111 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366561_6_non_const_i;
#line 1111 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366561_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 1112 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366561_6_non_const_i < num_histories)
#line 1113 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T2127;
 float __T2128;
 float __T2129;
 float __T2130;
 float __T2131;
 float __T2132;
 float __T2133;
 float __T2134;
 float __T2135;
 float __T2136;
 float __T2137;
 float __T2138;
 float __T2139;
#line 1114 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366564_10_non_const_rotation_angle_radians;
#line 1126 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366576_10_non_const_ut_entry_angle;
#line 1127 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366577_10_non_const_u_entry;
#line 1127 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366577_19_non_const_t_entry;
#line 1130 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_366580_8_non_const_entered;
#line 1140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366590_10_non_const_ut_exit_angle;
#line 1141 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366591_10_non_const_u_exit;
#line 1141 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366591_18_non_const_t_exit;
#line 1144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_366594_8_non_const_exited;
#line 1156 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366606_10_non_const_uv_entry_slope;
#line 1157 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366607_10_non_const_uv_exit_slope;
#line 1114 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366564_10_non_const_rotation_angle_radians = (((double)(gantry_angle[__cuda_local_var_366561_6_non_const_i])) * ( fdivide(((4.0) * (atan((1.0)))) , (180.0))));
#line 1126 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366576_10_non_const_ut_entry_angle = ((double)(((__T2127 = ((t_in_2[__cuda_local_var_366561_6_non_const_i]) - (t_in_1[__cuda_local_var_366561_6_non_const_i]))) , (void)(__T2128 = ((u_in_2[__cuda_local_var_366561_6_non_const_i]) - (u_in_1[__cuda_local_var_366561_6_non_const_i])))) , (atan2f(__T2127, __T2128))));
#line 1130 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366580_8_non_const_entered = (_Z20calculate_interceptsdddRdS_(((double)(u_in_2[__cuda_local_var_366561_6_non_const_i])), ((double)(t_in_2[__cuda_local_var_366561_6_non_const_i])), __cuda_local_var_366576_10_non_const_ut_entry_angle, (&__cuda_local_var_366577_10_non_const_u_entry), (&__cuda_local_var_366577_19_non_const_t_entry)));
#line 1132 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(xy_entry_angle[__cuda_local_var_366561_6_non_const_i]) = ((float)(__cuda_local_var_366576_10_non_const_ut_entry_angle + __cuda_local_var_366564_10_non_const_rotation_angle_radians));
#line 1135 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(x_entry[__cuda_local_var_366561_6_non_const_i]) = ((float)(((cos(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366577_10_non_const_u_entry) - ((sin(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366577_19_non_const_t_entry)));
#line 1136 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(y_entry[__cuda_local_var_366561_6_non_const_i]) = ((float)(((sin(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366577_10_non_const_u_entry) + ((cos(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366577_19_non_const_t_entry)));
#line 1140 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366590_10_non_const_ut_exit_angle = ((double)(((__T2129 = ((t_out_2[__cuda_local_var_366561_6_non_const_i]) - (t_out_1[__cuda_local_var_366561_6_non_const_i]))) , (void)(__T2130 = ((u_out_2[__cuda_local_var_366561_6_non_const_i]) - (u_out_1[__cuda_local_var_366561_6_non_const_i])))) , (atan2f(__T2129, __T2130))));
#line 1144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366594_8_non_const_exited = (_Z20calculate_interceptsdddRdS_(((double)(u_out_1[__cuda_local_var_366561_6_non_const_i])), ((double)(t_out_1[__cuda_local_var_366561_6_non_const_i])), __cuda_local_var_366590_10_non_const_ut_exit_angle, (&__cuda_local_var_366591_10_non_const_u_exit), (&__cuda_local_var_366591_18_non_const_t_exit)));
#line 1146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(xy_exit_angle[__cuda_local_var_366561_6_non_const_i]) = ((float)(__cuda_local_var_366590_10_non_const_ut_exit_angle + __cuda_local_var_366564_10_non_const_rotation_angle_radians));
#line 1149 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(x_exit[__cuda_local_var_366561_6_non_const_i]) = ((float)(((cos(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366591_10_non_const_u_exit) - ((sin(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366591_18_non_const_t_exit)));
#line 1150 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(y_exit[__cuda_local_var_366561_6_non_const_i]) = ((float)(((sin(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366591_10_non_const_u_exit) + ((cos(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * __cuda_local_var_366591_18_non_const_t_exit)));
#line 1156 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366606_10_non_const_uv_entry_slope = ((double)( fdividef(((v_in_2[__cuda_local_var_366561_6_non_const_i]) - (v_in_1[__cuda_local_var_366561_6_non_const_i])) , ((u_in_2[__cuda_local_var_366561_6_non_const_i]) - (u_in_1[__cuda_local_var_366561_6_non_const_i])))));
#line 1157 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366607_10_non_const_uv_exit_slope = ((double)( fdividef(((v_out_2[__cuda_local_var_366561_6_non_const_i]) - (v_out_1[__cuda_local_var_366561_6_non_const_i])) , ((u_out_2[__cuda_local_var_366561_6_non_const_i]) - (u_out_1[__cuda_local_var_366561_6_non_const_i])))));
#line 1159 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(xz_entry_angle[__cuda_local_var_366561_6_non_const_i]) = (((__T2131 = ((v_in_2[__cuda_local_var_366561_6_non_const_i]) - (v_in_1[__cuda_local_var_366561_6_non_const_i]))) , (void)(__T2132 = ((u_in_2[__cuda_local_var_366561_6_non_const_i]) - (u_in_1[__cuda_local_var_366561_6_non_const_i])))) , (atan2f(__T2131, __T2132)));
#line 1160 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(xz_exit_angle[__cuda_local_var_366561_6_non_const_i]) = (((__T2133 = ((v_out_2[__cuda_local_var_366561_6_non_const_i]) - (v_out_1[__cuda_local_var_366561_6_non_const_i]))) , (void)(__T2134 = ((u_out_2[__cuda_local_var_366561_6_non_const_i]) - (u_out_1[__cuda_local_var_366561_6_non_const_i])))) , (atan2f(__T2133, __T2134)));
#line 1169 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366577_10_non_const_u_entry = (((cos(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * ((double)(x_entry[__cuda_local_var_366561_6_non_const_i]))) + ((sin(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * ((double)(y_entry[__cuda_local_var_366561_6_non_const_i]))));
#line 1170 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366591_10_non_const_u_exit = (((cos(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * ((double)(x_exit[__cuda_local_var_366561_6_non_const_i]))) + ((sin(__cuda_local_var_366564_10_non_const_rotation_angle_radians)) * ((double)(y_exit[__cuda_local_var_366561_6_non_const_i]))));
#line 1171 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(z_entry[__cuda_local_var_366561_6_non_const_i]) = ((float)(((double)(v_in_2[__cuda_local_var_366561_6_non_const_i])) + (__cuda_local_var_366606_10_non_const_uv_entry_slope * (__cuda_local_var_366577_10_non_const_u_entry - ((double)(u_in_2[__cuda_local_var_366561_6_non_const_i]))))));
#line 1172 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(z_exit[__cuda_local_var_366561_6_non_const_i]) = ((float)(((double)(v_out_1[__cuda_local_var_366561_6_non_const_i])) - (__cuda_local_var_366607_10_non_const_uv_exit_slope * (((double)(u_out_1[__cuda_local_var_366561_6_non_const_i])) - __cuda_local_var_366591_10_non_const_u_exit))));
#line 1180 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_366580_8_non_const_entered) && (__cuda_local_var_366594_8_non_const_exited))
#line 1181 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1182 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((double)((__T2135 = (z_entry[__cuda_local_var_366561_6_non_const_i])) , (fabsf(__T2135)))) < (3.0)) && (((double)((__T2136 = (z_exit[__cuda_local_var_366561_6_non_const_i])) , (fabsf(__T2136)))) > (3.0)))
#line 1183 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2140;
#line 1184 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366634_12_non_const_recon_cyl_fraction;
#line 1184 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366634_12_non_const_recon_cyl_fraction = ((__T2140 = ( fdivide((((((double)(((int)((z_exit[__cuda_local_var_366561_6_non_const_i]) >= (0.0F))) - ((int)((z_exit[__cuda_local_var_366561_6_non_const_i]) < (0.0F))))) * (6.0)) * (0.5)) - ((double)(z_entry[__cuda_local_var_366561_6_non_const_i]))) , ((double)((z_exit[__cuda_local_var_366561_6_non_const_i]) - (z_entry[__cuda_local_var_366561_6_non_const_i])))))) , (fabs(__T2140)));
#line 1185 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(x_exit[__cuda_local_var_366561_6_non_const_i]) = ((float)(((double)(x_entry[__cuda_local_var_366561_6_non_const_i])) + (__cuda_local_var_366634_12_non_const_recon_cyl_fraction * ((double)((x_exit[__cuda_local_var_366561_6_non_const_i]) - (x_entry[__cuda_local_var_366561_6_non_const_i]))))));
#line 1186 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(y_exit[__cuda_local_var_366561_6_non_const_i]) = ((float)(((double)(y_entry[__cuda_local_var_366561_6_non_const_i])) + (__cuda_local_var_366634_12_non_const_recon_cyl_fraction * ((double)((y_exit[__cuda_local_var_366561_6_non_const_i]) - (y_entry[__cuda_local_var_366561_6_non_const_i]))))));
#line 1187 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(z_exit[__cuda_local_var_366561_6_non_const_i]) = ((float)((((double)(((int)((z_exit[__cuda_local_var_366561_6_non_const_i]) >= (0.0F))) - ((int)((z_exit[__cuda_local_var_366561_6_non_const_i]) < (0.0F))))) * (6.0)) * (0.5)));
#line 1188 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} else  {
#line 1189 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)((__T2137 = (z_entry[__cuda_local_var_366561_6_non_const_i])) , (fabsf(__T2137)))) > (3.0))
#line 1190 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1191 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366580_8_non_const_entered = ((__nv_bool)0);
#line 1192 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366594_8_non_const_exited = ((__nv_bool)0);
#line 1193 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 1198 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((__T2138 = ((t_out_1[__cuda_local_var_366561_6_non_const_i]) - (t_in_2[__cuda_local_var_366561_6_non_const_i]))) , (fabsf(__T2138))) > (5.0F)) || (((__T2139 = ((v_out_1[__cuda_local_var_366561_6_non_const_i]) - (v_in_2[__cuda_local_var_366561_6_non_const_i]))) , (fabsf(__T2139))) > (5.0F)))
#line 1199 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1200 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366580_8_non_const_entered = ((__nv_bool)0);
#line 1201 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366594_8_non_const_exited = ((__nv_bool)0);
#line 1202 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1203 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1206 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(missed_recon_volume[__cuda_local_var_366561_6_non_const_i]) = ((__nv_bool)((!(__cuda_local_var_366580_8_non_const_entered)) || (!(__cuda_local_var_366594_8_non_const_exited))));
#line 1207 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 1208 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1407 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z11binning_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(
#line 1409 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int num_histories, 
#line 1409 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_counts, 
#line 1409 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_num, 
#line 1409 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *missed_recon_volume, 
#line 1410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_entry, 
#line 1410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_entry, 
#line 1410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_entry, 
#line 1410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_exit, 
#line 1410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_exit, 
#line 1410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_exit, 
#line 1411 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_WEPL, 
#line 1411 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_ut_angle, 
#line 1411 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_uv_angle, 
#line 1411 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *WEPL, 
#line 1412 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_entry_angle, 
#line 1412 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_entry_angle, 
#line 1412 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_exit_angle, 
#line 1412 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_exit_angle){
#line 1414 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1415 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366865_6_non_const_i;
#line 1415 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366865_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 1416 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366865_6_non_const_i < num_histories)
#line 1417 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T2141;
 float __T2142;
#line 1423 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366873_11_non_const_x_midpath;
#line 1423 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366873_22_non_const_y_midpath;
#line 1423 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366873_33_non_const_z_midpath;
#line 1423 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366873_44_non_const_path_angle;
#line 1424 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366874_8_non_const_angle_bin;
#line 1424 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366874_19_non_const_t_bin;
#line 1424 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366874_26_non_const_v_bin;
#line 1425 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366875_11_non_const_angle;
#line 1425 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366875_18_non_const_t;
#line 1425 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366875_21_non_const_v;
#line 1426 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366876_11_non_const_rel_ut_angle;
#line 1426 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_366876_25_non_const_rel_uv_angle;
#line 1429 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366873_11_non_const_x_midpath = ((double)( fdividef(((x_entry[__cuda_local_var_366865_6_non_const_i]) + (x_exit[__cuda_local_var_366865_6_non_const_i])) , (2.0F))));
#line 1430 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366873_22_non_const_y_midpath = ((double)( fdividef(((y_entry[__cuda_local_var_366865_6_non_const_i]) + (y_exit[__cuda_local_var_366865_6_non_const_i])) , (2.0F))));
#line 1431 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366873_33_non_const_z_midpath = ((double)( fdividef(((z_entry[__cuda_local_var_366865_6_non_const_i]) + (z_exit[__cuda_local_var_366865_6_non_const_i])) , (2.0F))));
#line 1434 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366873_44_non_const_path_angle = ((double)(((__T2141 = ((y_exit[__cuda_local_var_366865_6_non_const_i]) - (y_entry[__cuda_local_var_366865_6_non_const_i]))) , (void)(__T2142 = ((x_exit[__cuda_local_var_366865_6_non_const_i]) - (x_entry[__cuda_local_var_366865_6_non_const_i])))) , (atan2f(__T2141, __T2142))));
#line 1435 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366873_44_non_const_path_angle < (0.0)) {
#line 1436 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366873_44_non_const_path_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1437 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366874_8_non_const_angle_bin = (((int)__double2int_rz((double)((( fdivide((__cuda_local_var_366873_44_non_const_path_angle * ( fdivide((180.0) , ((4.0) * (atan((1.0))))))) , (6.0))) + (0.5))))) % 60);
#line 1438 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366875_11_non_const_angle = ((((double)__cuda_local_var_366874_8_non_const_angle_bin) * (6.0)) * ( fdivide(((4.0) * (atan((1.0)))) , (180.0))));
#line 1441 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366875_18_non_const_t = ((__cuda_local_var_366873_22_non_const_y_midpath * (cos(__cuda_local_var_366875_11_non_const_angle))) - (__cuda_local_var_366873_11_non_const_x_midpath * (sin(__cuda_local_var_366875_11_non_const_angle))));
#line 1442 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366874_19_non_const_t_bin = ((int)__double2int_rz((double)((( fdivide(__cuda_local_var_366875_18_non_const_t , (0.10000000000000001))) + (175.0)))));
#line 1444 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366875_21_non_const_v = __cuda_local_var_366873_33_non_const_z_midpath;
#line 1445 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366874_26_non_const_v_bin = ((int)__double2int_rz((double)((( fdivide(__cuda_local_var_366875_21_non_const_v , (0.25))) + (18.0)))));
#line 1448 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((__cuda_local_var_366874_19_non_const_t_bin >= 0) && (__cuda_local_var_366874_26_non_const_v_bin >= 0)) && (__cuda_local_var_366874_19_non_const_t_bin < 350)) && (__cuda_local_var_366874_26_non_const_v_bin < 36))
#line 1449 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1450 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(bin_num[__cuda_local_var_366865_6_non_const_i]) = ((__cuda_local_var_366874_19_non_const_t_bin + (__cuda_local_var_366874_8_non_const_angle_bin * 350)) + ((__cuda_local_var_366874_26_non_const_v_bin * 350) * 60));
#line 1451 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(missed_recon_volume[__cuda_local_var_366865_6_non_const_i]))
#line 1452 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1457 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366876_11_non_const_rel_ut_angle = ((double)((xy_exit_angle[__cuda_local_var_366865_6_non_const_i]) - (xy_entry_angle[__cuda_local_var_366865_6_non_const_i])));
#line 1458 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366876_11_non_const_rel_ut_angle > ((4.0) * (atan((1.0))))) {
#line 1459 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366876_11_non_const_rel_ut_angle -= ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1460 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366876_11_non_const_rel_ut_angle < (-((4.0) * (atan((1.0)))))) {
#line 1461 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366876_11_non_const_rel_ut_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1462 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366876_25_non_const_rel_uv_angle = ((double)((xz_exit_angle[__cuda_local_var_366865_6_non_const_i]) - (xz_entry_angle[__cuda_local_var_366865_6_non_const_i])));
#line 1463 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366876_25_non_const_rel_uv_angle > ((4.0) * (atan((1.0))))) {
#line 1464 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366876_25_non_const_rel_uv_angle -= ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1465 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_366876_25_non_const_rel_uv_angle < (-((4.0) * (atan((1.0)))))) {
#line 1466 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366876_25_non_const_rel_uv_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1467 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((bin_counts + (bin_num[__cuda_local_var_366865_6_non_const_i])), 1);
#line 1468 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((mean_WEPL + (bin_num[__cuda_local_var_366865_6_non_const_i])), (WEPL[__cuda_local_var_366865_6_non_const_i]));
#line 1469 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((mean_rel_ut_angle + (bin_num[__cuda_local_var_366865_6_non_const_i])), ((float)__cuda_local_var_366876_11_non_const_rel_ut_angle));
#line 1470 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((mean_rel_ut_angle + (bin_num[__cuda_local_var_366865_6_non_const_i])), ((float)__cuda_local_var_366876_25_non_const_rel_uv_angle));
#line 1473 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} else  {
#line 1475 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(bin_num[__cuda_local_var_366865_6_non_const_i]) = (-1); }
#line 1476 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1477 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 1478 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z19calculate_means_GPUPiPfS0_S0_(
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_counts, 
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_WEPL, 
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_ut_angle, 
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_uv_angle){
#line 1515 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366966_6_non_const_v;
#line 1516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366966_22_non_const_angle;
#line 1516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366966_42_non_const_t;
#line 1517 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_366967_6_non_const_bin;
#line 1516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366966_6_non_const_v = ((int)(blockIdx.x));
#line 1516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366966_22_non_const_angle = ((int)(blockIdx.y));
#line 1516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366966_42_non_const_t = ((int)(threadIdx.x));
#line 1517 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_366967_6_non_const_bin = ((__cuda_local_var_366966_42_non_const_t + (__cuda_local_var_366966_22_non_const_angle * 350)) + ((__cuda_local_var_366966_6_non_const_v * 350) * 60));
#line 1518 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((bin_counts[__cuda_local_var_366967_6_non_const_bin]) > 0)
#line 1519 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1520 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(mean_WEPL[__cuda_local_var_366967_6_non_const_bin]) = ( fdividef((mean_WEPL[__cuda_local_var_366967_6_non_const_bin]) , ((float)(bin_counts[__cuda_local_var_366967_6_non_const_bin]))));
#line 1521 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(mean_rel_ut_angle[__cuda_local_var_366967_6_non_const_bin]) = ( fdividef((mean_rel_ut_angle[__cuda_local_var_366967_6_non_const_bin]) , ((float)(bin_counts[__cuda_local_var_366967_6_non_const_bin]))));
#line 1522 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(mean_rel_uv_angle[__cuda_local_var_366967_6_non_const_bin]) = ( fdividef((mean_rel_uv_angle[__cuda_local_var_366967_6_non_const_bin]) , ((float)(bin_counts[__cuda_local_var_366967_6_non_const_bin]))));
#line 1523 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 1524 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1568 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z26sum_squared_deviations_GPUiPiPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_(
#line 1570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int num_histories, 
#line 1570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_num, 
#line 1570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_WEPL, 
#line 1570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_ut_angle, 
#line 1570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_uv_angle, 
#line 1571 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *WEPL, 
#line 1571 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_entry_angle, 
#line 1571 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_entry_angle, 
#line 1571 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_exit_angle, 
#line 1571 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_exit_angle, 
#line 1572 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_WEPL, 
#line 1572 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_rel_ut_angle, 
#line 1572 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_rel_uv_angle){
#line 1574 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1575 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367025_9_non_const_WEPL_difference;
#line 1575 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367025_26_non_const_rel_ut_angle_difference;
#line 1575 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367025_51_non_const_rel_uv_angle_difference;
#line 1576 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367026_6_non_const_i;
#line 1576 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367026_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 1577 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367026_6_non_const_i < num_histories)
#line 1578 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1579 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367029_10_non_const_rel_ut_angle;
#line 1584 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367034_10_non_const_rel_uv_angle;
#line 1579 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367029_10_non_const_rel_ut_angle = ((double)((xy_exit_angle[__cuda_local_var_367026_6_non_const_i]) - (xy_entry_angle[__cuda_local_var_367026_6_non_const_i])));
#line 1580 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367029_10_non_const_rel_ut_angle > ((4.0) * (atan((1.0))))) {
#line 1581 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367029_10_non_const_rel_ut_angle -= ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1582 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367029_10_non_const_rel_ut_angle < (-((4.0) * (atan((1.0)))))) {
#line 1583 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367029_10_non_const_rel_ut_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1584 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367034_10_non_const_rel_uv_angle = ((double)((xz_exit_angle[__cuda_local_var_367026_6_non_const_i]) - (xz_entry_angle[__cuda_local_var_367026_6_non_const_i])));
#line 1585 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367034_10_non_const_rel_uv_angle > ((4.0) * (atan((1.0))))) {
#line 1586 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367034_10_non_const_rel_uv_angle -= ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1587 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367034_10_non_const_rel_uv_angle < (-((4.0) * (atan((1.0)))))) {
#line 1588 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367034_10_non_const_rel_uv_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1589 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367025_9_non_const_WEPL_difference = ((double)((WEPL[__cuda_local_var_367026_6_non_const_i]) - (mean_WEPL[(bin_num[__cuda_local_var_367026_6_non_const_i])])));
#line 1590 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367025_26_non_const_rel_ut_angle_difference = (__cuda_local_var_367029_10_non_const_rel_ut_angle - ((double)(mean_rel_ut_angle[(bin_num[__cuda_local_var_367026_6_non_const_i])])));
#line 1591 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367025_51_non_const_rel_uv_angle_difference = (__cuda_local_var_367034_10_non_const_rel_uv_angle - ((double)(mean_rel_uv_angle[(bin_num[__cuda_local_var_367026_6_non_const_i])])));
#line 1593 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((stddev_WEPL + (bin_num[__cuda_local_var_367026_6_non_const_i])), ((float)(_Z8_Pow_intIdET_S0_i(__cuda_local_var_367025_9_non_const_WEPL_difference, 2))));
#line 1594 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((stddev_rel_ut_angle + (bin_num[__cuda_local_var_367026_6_non_const_i])), ((float)(_Z8_Pow_intIdET_S0_i(__cuda_local_var_367025_26_non_const_rel_ut_angle_difference, 2))));
#line 1595 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((stddev_rel_uv_angle + (bin_num[__cuda_local_var_367026_6_non_const_i])), ((float)(_Z8_Pow_intIdET_S0_i(__cuda_local_var_367025_51_non_const_rel_uv_angle_difference, 2))));
#line 1596 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 1597 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z33calculate_standard_deviations_GPUPiPfS0_S0_(
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_counts, 
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_WEPL, 
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_rel_ut_angle, 
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_rel_uv_angle){
#line 1610 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1611 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367061_6_non_const_v;
#line 1611 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367061_22_non_const_angle;
#line 1611 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367061_42_non_const_t;
#line 1612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367062_6_non_const_bin;
#line 1611 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367061_6_non_const_v = ((int)(blockIdx.x));
#line 1611 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367061_22_non_const_angle = ((int)(blockIdx.y));
#line 1611 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367061_42_non_const_t = ((int)(threadIdx.x));
#line 1612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367062_6_non_const_bin = ((__cuda_local_var_367061_42_non_const_t + (__cuda_local_var_367061_22_non_const_angle * 350)) + ((__cuda_local_var_367061_6_non_const_v * 350) * 60));
#line 1613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((bin_counts[__cuda_local_var_367062_6_non_const_bin]) > 0)
#line 1614 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1616 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(stddev_WEPL[__cuda_local_var_367062_6_non_const_bin]) = (sqrtf(( fdividef((stddev_WEPL[__cuda_local_var_367062_6_non_const_bin]) , ((float)((bin_counts[__cuda_local_var_367062_6_non_const_bin]) - 1))))));
#line 1617 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(stddev_rel_ut_angle[__cuda_local_var_367062_6_non_const_bin]) = (sqrtf(( fdividef((stddev_rel_ut_angle[__cuda_local_var_367062_6_non_const_bin]) , ((float)((bin_counts[__cuda_local_var_367062_6_non_const_bin]) - 1))))));
#line 1618 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(stddev_rel_uv_angle[__cuda_local_var_367062_6_non_const_bin]) = (sqrtf(( fdividef((stddev_rel_uv_angle[__cuda_local_var_367062_6_non_const_bin]) , ((float)((bin_counts[__cuda_local_var_367062_6_non_const_bin]) - 1))))));
#line 1619 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1620 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"
__syncthreads();
#line 1620 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1621 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(bin_counts[__cuda_local_var_367062_6_non_const_bin]) = 0; 
#line 1622 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1702 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z20statistical_cuts_GPUiPiS_PfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_Pb(
#line 1704 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int num_histories, 
#line 1704 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_counts, 
#line 1704 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_num, 
#line 1704 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *sinogram, 
#line 1704 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *WEPL, 
#line 1705 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_entry_angle, 
#line 1705 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_entry_angle, 
#line 1705 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xy_exit_angle, 
#line 1705 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *xz_exit_angle, 
#line 1706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_WEPL, 
#line 1706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_ut_angle, 
#line 1706 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *mean_rel_uv_angle, 
#line 1707 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_WEPL, 
#line 1707 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_rel_ut_angle, 
#line 1707 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *stddev_rel_uv_angle, 
#line 1708 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *failed_cuts){
#line 1710 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1711 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367161_6_non_const_i;
#line 1711 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367161_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 1712 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367161_6_non_const_i < num_histories)
#line 1713 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2143;
 double __T2144;
 float __T2145;
#line 1714 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367164_10_non_const_rel_ut_angle;
#line 1719 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367169_10_non_const_rel_uv_angle;
#line 1724 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367174_8_non_const_passed_ut_cut;
#line 1725 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367175_8_non_const_passed_uv_cut;
#line 1726 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367176_8_non_const_passed_WEPL_cut;
#line 1714 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367164_10_non_const_rel_ut_angle = ((double)((xy_exit_angle[__cuda_local_var_367161_6_non_const_i]) - (xy_entry_angle[__cuda_local_var_367161_6_non_const_i])));
#line 1715 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367164_10_non_const_rel_ut_angle > ((4.0) * (atan((1.0))))) {
#line 1716 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367164_10_non_const_rel_ut_angle -= ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1717 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367164_10_non_const_rel_ut_angle < (-((4.0) * (atan((1.0)))))) {
#line 1718 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367164_10_non_const_rel_ut_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1719 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367169_10_non_const_rel_uv_angle = ((double)((xz_exit_angle[__cuda_local_var_367161_6_non_const_i]) - (xz_entry_angle[__cuda_local_var_367161_6_non_const_i])));
#line 1720 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367169_10_non_const_rel_uv_angle > ((4.0) * (atan((1.0))))) {
#line 1721 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367169_10_non_const_rel_uv_angle -= ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1722 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367169_10_non_const_rel_uv_angle < (-((4.0) * (atan((1.0)))))) {
#line 1723 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367169_10_non_const_rel_uv_angle += ((2.0) * ((4.0) * (atan((1.0))))); }
#line 1724 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367174_8_non_const_passed_ut_cut = ((__nv_bool)(((__T2143 = (__cuda_local_var_367164_10_non_const_rel_ut_angle - ((double)(mean_rel_ut_angle[(bin_num[__cuda_local_var_367161_6_non_const_i])])))) , (fabs(__T2143))) < ((double)((3.0F) * (stddev_rel_ut_angle[(bin_num[__cuda_local_var_367161_6_non_const_i])])))));
#line 1725 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367175_8_non_const_passed_uv_cut = ((__nv_bool)(((__T2144 = (__cuda_local_var_367169_10_non_const_rel_uv_angle - ((double)(mean_rel_uv_angle[(bin_num[__cuda_local_var_367161_6_non_const_i])])))) , (fabs(__T2144))) < ((double)((3.0F) * (stddev_rel_uv_angle[(bin_num[__cuda_local_var_367161_6_non_const_i])])))));
#line 1726 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367176_8_non_const_passed_WEPL_cut = ((__nv_bool)(((__T2145 = ((mean_WEPL[(bin_num[__cuda_local_var_367161_6_non_const_i])]) - (WEPL[__cuda_local_var_367161_6_non_const_i]))) , (fabsf(__T2145))) <= ((3.0F) * (stddev_WEPL[(bin_num[__cuda_local_var_367161_6_non_const_i])]))));
#line 1727 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(failed_cuts[__cuda_local_var_367161_6_non_const_i]) = ((__nv_bool)(((!(__cuda_local_var_367174_8_non_const_passed_ut_cut)) || (!(__cuda_local_var_367175_8_non_const_passed_uv_cut))) || (!(__cuda_local_var_367176_8_non_const_passed_WEPL_cut))));
#line 1729 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(failed_cuts[__cuda_local_var_367161_6_non_const_i]))
#line 1730 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1731 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((bin_counts + (bin_num[__cuda_local_var_367161_6_non_const_i])), 1);
#line 1732 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPff((sinogram + (bin_num[__cuda_local_var_367161_6_non_const_i])), (WEPL[__cuda_local_var_367161_6_non_const_i]));
#line 1733 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1734 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 1735 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1766 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z22construct_sinogram_GPUPiPf(
#line 1766 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_counts, 
#line 1766 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *sinogram){
#line 1767 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1768 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367218_6_non_const_v;
#line 1768 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367218_22_non_const_angle;
#line 1768 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367218_42_non_const_t;
#line 1769 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367219_6_non_const_bin;
#line 1768 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367218_6_non_const_v = ((int)(blockIdx.x));
#line 1768 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367218_22_non_const_angle = ((int)(blockIdx.y));
#line 1768 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367218_42_non_const_t = ((int)(threadIdx.x));
#line 1769 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367219_6_non_const_bin = ((__cuda_local_var_367218_42_non_const_t + (__cuda_local_var_367218_22_non_const_angle * 350)) + ((__cuda_local_var_367218_6_non_const_v * 350) * 60));
#line 1770 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((bin_counts[__cuda_local_var_367219_6_non_const_bin]) > 0) {
#line 1771 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(sinogram[__cuda_local_var_367219_6_non_const_bin]) = ( fdividef((sinogram[__cuda_local_var_367219_6_non_const_bin]) , ((float)(bin_counts[__cuda_local_var_367219_6_non_const_bin])))); } 
#line 1772 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1839 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z10filter_GPUPfS_(
#line 1839 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *sinogram, 
#line 1839 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *sinogram_filtered){
#line 1840 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1841 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367291_6_non_const_v_bin;
#line 1841 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367291_26_non_const_angle_bin;
#line 1841 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367291_50_non_const_t_bin;
#line 1842 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367292_6_non_const_t_bin_ref;
#line 1842 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367292_17_non_const_t_bin_sep;
#line 1842 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367292_28_non_const_strip_index;
#line 1843 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367293_9_non_const_filtered;
#line 1843 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367293_19_non_const_t;
#line 1843 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367293_22_non_const_scale_factor;
#line 1844 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367294_9_non_const_v;
#line 1841 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367291_6_non_const_v_bin = ((int)(blockIdx.x));
#line 1841 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367291_26_non_const_angle_bin = ((int)(blockIdx.y));
#line 1841 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367291_50_non_const_t_bin = ((int)(threadIdx.x));
#line 1844 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367294_9_non_const_v = ((((double)(__cuda_local_var_367291_6_non_const_v_bin - 18)) * (0.25)) + (0.125));
#line 1847 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (__cuda_local_var_367292_6_non_const_t_bin_ref = 0; (__cuda_local_var_367292_6_non_const_t_bin_ref < 350); __cuda_local_var_367292_6_non_const_t_bin_ref++)
#line 1848 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1849 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367293_19_non_const_t = ((((double)(__cuda_local_var_367292_6_non_const_t_bin_ref - 175)) * (0.10000000000000001)) + (0.050000000000000003));
#line 1850 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367292_17_non_const_t_bin_sep = (__cuda_local_var_367291_50_non_const_t_bin - __cuda_local_var_367292_6_non_const_t_bin_ref);
#line 1852 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367293_22_non_const_scale_factor = ( fdivide((265.69999999999999) , (sqrt((((70596.489999999991) + (__cuda_local_var_367293_19_non_const_t * __cuda_local_var_367293_19_non_const_t)) + (__cuda_local_var_367294_9_non_const_v * __cuda_local_var_367294_9_non_const_v))))));
#line 1853 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
switch (1)
#line 1854 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1855 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
case 2:
#line 1856 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
goto __T2146;
#line 1857 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
case 0:
#line 1858 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367292_17_non_const_t_bin_sep == 0) {
#line 1859 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367293_9_non_const_filtered = ((1.0) / ((4.0) * (pow((((double)( fdividef((2.0F) , (sqrtf((2.0F)))))) * (0.10000000000000001)), (2.0))))); } else  {
#line 1860 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_367292_17_non_const_t_bin_sep % 2) == 0) {
#line 1861 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367293_9_non_const_filtered = (0.0); } else  {
#line 1863 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367293_9_non_const_filtered = ( fdivide((-1.0) , (pow((((((double)( fdividef((2.0F) , (sqrtf((2.0F)))))) * (0.10000000000000001)) * ((4.0) * (atan((1.0))))) * ((double)__cuda_local_var_367292_17_non_const_t_bin_sep)), (2.0))))); } }
#line 1864 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
goto __T2146;
#line 1865 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
case 1:
#line 1866 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367293_9_non_const_filtered = (pow(((pow(((0.10000000000000001) * ((4.0) * (atan((1.0))))), (2.0))) * ((1.0) - (pow(((double)(2 * __cuda_local_var_367292_17_non_const_t_bin_sep)), (2.0))))), (-1.0)));
#line 1867 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} __T2146:;
#line 1868 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367292_28_non_const_strip_index = (((__cuda_local_var_367291_6_non_const_v_bin * 60) * 350) + (__cuda_local_var_367291_26_non_const_angle_bin * 350));
#line 1869 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(sinogram_filtered[(__cuda_local_var_367292_28_non_const_strip_index + __cuda_local_var_367291_50_non_const_t_bin)]) += ((((0.10000000000000001) * ((double)(sinogram[(__cuda_local_var_367292_28_non_const_strip_index + __cuda_local_var_367292_6_non_const_t_bin_ref)]))) * __cuda_local_var_367293_9_non_const_filtered) * __cuda_local_var_367293_22_non_const_scale_factor);
#line 1870 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 1871 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 1957 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z18backprojection_GPUPfS_(
#line 1957 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *sinogram_filtered, 
#line 1957 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *FBP_image){
#line 1958 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1959 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367409_6_non_const_row;
#line 1959 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367409_24_non_const_column;
#line 1959 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367409_45_non_const_slice;
#line 1960 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367410_6_non_const_voxel;
#line 1959 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367409_6_non_const_row = ((int)(blockIdx.y));
#line 1959 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367409_24_non_const_column = ((int)(blockIdx.x));
#line 1959 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367409_45_non_const_slice = ((int)(threadIdx.x));
#line 1960 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367410_6_non_const_voxel = ((((__cuda_local_var_367409_45_non_const_slice * 200) * 200) + (__cuda_local_var_367409_6_non_const_row * 200)) + __cuda_local_var_367409_24_non_const_column);
#line 1961 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367410_6_non_const_voxel < 960000)
#line 1962 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 1963 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367413_10_non_const_delta;
#line 1964 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367414_10_non_const_u;
#line 1964 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367414_13_non_const_t;
#line 1964 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367414_16_non_const_v;
#line 1965 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367415_10_non_const_detector_number_t;
#line 1965 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367415_29_non_const_detector_number_v;
#line 1966 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367416_10_non_const_eta;
#line 1966 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367416_15_non_const_epsilon;
#line 1967 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367417_10_non_const_scale_factor;
#line 1968 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367418_7_non_const_t_bin;
#line 1968 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367418_14_non_const_v_bin;
#line 1968 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367418_21_non_const_bin;
#line 1969 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367419_10_non_const_x;
#line 1970 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367420_10_non_const_y;
#line 1971 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367421_10_non_const_z;
#line 1963 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367413_10_non_const_delta = ((6.0) * ( fdivide(((4.0) * (atan((1.0)))) , (180.0))));
#line 1969 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367419_10_non_const_x = ((-8.0) + ((((double)__cuda_local_var_367409_24_non_const_column) + (0.5)) * (0.080000000000000002)));
#line 1970 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367420_10_non_const_y = ((8.0) - ((((double)__cuda_local_var_367409_6_non_const_row) + (0.5)) * (0.080000000000000002)));
#line 1971 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367421_10_non_const_z = ((-3.0) + ((((double)__cuda_local_var_367409_45_non_const_slice) + (0.5)) * (0.25)));
#line 1974 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367419_10_non_const_x * __cuda_local_var_367419_10_non_const_x) + (__cuda_local_var_367420_10_non_const_y * __cuda_local_var_367420_10_non_const_y)) > (64.0)) {
#line 1975 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(FBP_image[((((__cuda_local_var_367409_45_non_const_slice * 200) * 200) + (__cuda_local_var_367409_6_non_const_row * 200)) + __cuda_local_var_367409_24_non_const_column)]) = (0.00112999999F); }
#line 1977 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 1977 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{ {
#line 1979 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int angle_bin;
#line 1979 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
angle_bin = 0;
#line 1979 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (angle_bin < 60); angle_bin++)
#line 1980 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2147;
#line 1982 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367414_10_non_const_u = ((__cuda_local_var_367419_10_non_const_x * (cos((((double)angle_bin) * __cuda_local_var_367413_10_non_const_delta)))) + (__cuda_local_var_367420_10_non_const_y * (sin((((double)angle_bin) * __cuda_local_var_367413_10_non_const_delta)))));
#line 1983 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367414_13_non_const_t = (((-__cuda_local_var_367419_10_non_const_x) * (sin((((double)angle_bin) * __cuda_local_var_367413_10_non_const_delta)))) + (__cuda_local_var_367420_10_non_const_y * (cos((((double)angle_bin) * __cuda_local_var_367413_10_non_const_delta)))));
#line 1984 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367414_16_non_const_v = __cuda_local_var_367421_10_non_const_z;
#line 1987 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367415_10_non_const_detector_number_t = (( fdivide((__cuda_local_var_367414_13_non_const_t - (__cuda_local_var_367414_10_non_const_u * ( fdivide(__cuda_local_var_367414_13_non_const_t , ((265.69999999999999) + __cuda_local_var_367414_10_non_const_u))))) , (0.10000000000000001))) + (175.0));
#line 1988 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367418_7_non_const_t_bin = ((int)__double2int_rz((double)(__cuda_local_var_367415_10_non_const_detector_number_t)));
#line 1989 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)__cuda_local_var_367418_7_non_const_t_bin) > __cuda_local_var_367415_10_non_const_detector_number_t) {
#line 1990 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367418_7_non_const_t_bin -= 1; }
#line 1991 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367416_10_non_const_eta = (__cuda_local_var_367415_10_non_const_detector_number_t - ((double)__cuda_local_var_367418_7_non_const_t_bin));
#line 1994 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367415_29_non_const_detector_number_v = (( fdivide((__cuda_local_var_367414_16_non_const_v - (__cuda_local_var_367414_10_non_const_u * ( fdivide(__cuda_local_var_367414_16_non_const_v , ((265.69999999999999) + __cuda_local_var_367414_10_non_const_u))))) , (0.25))) + (18.0));
#line 1995 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367418_14_non_const_v_bin = ((int)__double2int_rz((double)(__cuda_local_var_367415_29_non_const_detector_number_v)));
#line 1996 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)__cuda_local_var_367418_14_non_const_v_bin) > __cuda_local_var_367415_29_non_const_detector_number_v) {
#line 1997 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367418_14_non_const_v_bin -= 1; }
#line 1998 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367416_15_non_const_epsilon = (__cuda_local_var_367415_29_non_const_detector_number_v - ((double)__cuda_local_var_367418_14_non_const_v_bin));
#line 2001 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367417_10_non_const_scale_factor = ((__T2147 = ( fdivide((265.69999999999999) , ((265.69999999999999) + __cuda_local_var_367414_10_non_const_u)))) , (_Z8_Pow_intIdET_S0_i(__T2147, 2)));
#line 2005 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367418_21_non_const_bin = ((__cuda_local_var_367418_7_non_const_t_bin + (angle_bin * 350)) + ((__cuda_local_var_367418_14_non_const_v_bin * 60) * 350));
#line 2010 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_367418_14_non_const_v_bin == 35) || (__cuda_local_var_367418_21_non_const_bin < 0)) {
#line 2011 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(FBP_image[__cuda_local_var_367410_6_non_const_voxel]) += (__cuda_local_var_367417_10_non_const_scale_factor * ((((1.0) - __cuda_local_var_367416_10_non_const_eta) * ((double)(sinogram_filtered[__cuda_local_var_367418_21_non_const_bin]))) + (__cuda_local_var_367416_10_non_const_eta * ((double)(sinogram_filtered[(__cuda_local_var_367418_21_non_const_bin + 1)]))))); }
#line 2014 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2014 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2023 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(FBP_image[__cuda_local_var_367410_6_non_const_voxel]) += (__cuda_local_var_367417_10_non_const_scale_factor * (((((((1.0) - __cuda_local_var_367416_10_non_const_eta) * ((1.0) - __cuda_local_var_367416_15_non_const_epsilon)) * ((double)(sinogram_filtered[__cuda_local_var_367418_21_non_const_bin]))) + ((__cuda_local_var_367416_10_non_const_eta * ((1.0) - __cuda_local_var_367416_15_non_const_epsilon)) * ((double)(sinogram_filtered[(__cuda_local_var_367418_21_non_const_bin + 1)])))) + ((((1.0) - __cuda_local_var_367416_10_non_const_eta) * __cuda_local_var_367416_15_non_const_epsilon) * ((double)(sinogram_filtered[(__cuda_local_var_367418_21_non_const_bin + 21000)])))) + ((__cuda_local_var_367416_10_non_const_eta * __cuda_local_var_367416_15_non_const_epsilon) * ((double)(sinogram_filtered[((__cuda_local_var_367418_21_non_const_bin + 21000) + 1)])))));
#line 2027 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2028 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2029 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(FBP_image[__cuda_local_var_367410_6_non_const_voxel]) *= __cuda_local_var_367413_10_non_const_delta;
#line 2030 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2031 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 2032 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2080 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z20FBP_image_2_hull_GPUPfPb(
#line 2080 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *FBP_image, 
#line 2080 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *FBP_hull){
#line 2081 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367532_6_non_const_row;
#line 2082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367532_24_non_const_column;
#line 2082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367532_45_non_const_slice;
#line 2083 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367533_6_non_const_voxel;
#line 2084 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367534_9_non_const_x;
#line 2085 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367535_9_non_const_y;
#line 2086 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367536_9_non_const_d_squared;
#line 2082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367532_6_non_const_row = ((int)(blockIdx.y));
#line 2082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367532_24_non_const_column = ((int)(blockIdx.x));
#line 2082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367532_45_non_const_slice = ((int)(threadIdx.x));
#line 2083 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367533_6_non_const_voxel = ((((__cuda_local_var_367532_45_non_const_slice * 200) * 200) + (__cuda_local_var_367532_6_non_const_row * 200)) + __cuda_local_var_367532_24_non_const_column);
#line 2084 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367534_9_non_const_x = ((-8.0) + ((((double)__cuda_local_var_367532_24_non_const_column) + (0.5)) * (0.080000000000000002)));
#line 2085 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367535_9_non_const_y = ((8.0) - ((((double)__cuda_local_var_367532_6_non_const_row) + (0.5)) * (0.080000000000000002)));
#line 2086 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367536_9_non_const_d_squared = ((_Z8_Pow_intIdET_S0_i(__cuda_local_var_367534_9_non_const_x, 2)) + (_Z8_Pow_intIdET_S0_i(__cuda_local_var_367535_9_non_const_y, 2)));
#line 2087 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((double)(FBP_image[__cuda_local_var_367533_6_non_const_voxel])) > (0.59999999999999998)) && (__cuda_local_var_367536_9_non_const_d_squared < (_Z8_Pow_intIdET_S0_i((8.0), 2)))) {
#line 2088 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(FBP_hull[__cuda_local_var_367533_6_non_const_voxel]) = ((__nv_bool)1); } else  {
#line 2090 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(FBP_hull[__cuda_local_var_367533_6_non_const_voxel]) = ((__nv_bool)0); } 
#line 2091 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2331 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z17carve_differencesPiS_(
#line 2331 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *carve_differences, 
#line 2331 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *image){
#line 2332 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2333 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367783_6_non_const_row;
#line 2333 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367783_24_non_const_column;
#line 2333 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367783_45_non_const_slice;
#line 2334 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367784_6_non_const_voxel;
#line 2333 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367783_6_non_const_row = ((int)(blockIdx.y));
#line 2333 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367783_24_non_const_column = ((int)(blockIdx.x));
#line 2333 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367783_45_non_const_slice = ((int)(threadIdx.x));
#line 2334 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367784_6_non_const_voxel = ((__cuda_local_var_367783_24_non_const_column + (__cuda_local_var_367783_6_non_const_row * 200)) + ((__cuda_local_var_367783_45_non_const_slice * 200) * 200));
#line 2335 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((__cuda_local_var_367783_6_non_const_row != 0) && (__cuda_local_var_367783_6_non_const_row != 199)) && (__cuda_local_var_367783_24_non_const_column != 0)) && (__cuda_local_var_367783_24_non_const_column != 199))
#line 2336 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2337 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367787_7_non_const_difference;
#line 2337 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367787_19_non_const_max_difference;
#line 2337 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367787_19_non_const_max_difference = 0; {
#line 2338 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int current_row;
#line 2338 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
current_row = (__cuda_local_var_367783_6_non_const_row - 1);
#line 2338 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (current_row <= (__cuda_local_var_367783_6_non_const_row + 1)); current_row++)
#line 2339 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{ {
#line 2340 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int current_column;
#line 2340 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
current_column = (__cuda_local_var_367783_24_non_const_column - 1);
#line 2340 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (current_column <= (__cuda_local_var_367783_24_non_const_column + 1)); current_column++)
#line 2341 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2342 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367787_7_non_const_difference = ((image[__cuda_local_var_367784_6_non_const_voxel]) - (image[((current_column + (current_row * 200)) + ((__cuda_local_var_367783_45_non_const_slice * 200) * 200))]));
#line 2343 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367787_7_non_const_difference > __cuda_local_var_367787_19_non_const_max_difference) {
#line 2344 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367787_19_non_const_max_difference = __cuda_local_var_367787_7_non_const_difference; }
#line 2345 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2346 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2347 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(carve_differences[__cuda_local_var_367784_6_non_const_voxel]) = __cuda_local_var_367787_19_non_const_max_difference;
#line 2348 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 2349 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2364 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z6SC_GPUiPbPiS_PfS1_S1_S1_S1_S1_S1_(
#line 2366 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
const int num_histories, 
#line 2366 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *SC_hull, 
#line 2366 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_num, 
#line 2366 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *missed_recon_volume, 
#line 2366 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *WEPL, 
#line 2367 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_entry, 
#line 2367 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_entry, 
#line 2367 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_entry, 
#line 2367 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_exit, 
#line 2367 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_exit, 
#line 2367 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_exit){
#line 2369 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2370 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367820_6_non_const_i;
#line 2372 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367822_7_non_const_run_it;
#line 2370 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367820_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 2372 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367822_7_non_const_run_it = ((__nv_bool)(__cuda_local_var_367820_6_non_const_i == 192644));
#line 2373 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((__cuda_local_var_367822_7_non_const_run_it) && (__cuda_local_var_367820_6_non_const_i < num_histories)) && (!(missed_recon_volume[__cuda_local_var_367820_6_non_const_i]))) && (((double)(WEPL[__cuda_local_var_367820_6_non_const_i])) <= (0.0)))
#line 2374 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T2148;
 float __T2149;
 float __T2150;
 float __T2151;
 float __T2152;
 float __T2153;
#line 2378 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367828_7_non_const_x_move_direction;
#line 2378 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367828_25_non_const_y_move_direction;
#line 2378 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367828_43_non_const_z_move_direction;
#line 2379 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367829_10_non_const_delta_yx;
#line 2379 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367829_20_non_const_delta_zx;
#line 2379 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367829_30_non_const_delta_zy;
#line 2380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367830_10_non_const_x_move;
#line 2380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367830_22_non_const_y_move;
#line 2380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367830_34_non_const_z_move;
#line 2385 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367835_10_non_const_x_to_go;
#line 2385 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367835_19_non_const_y_to_go;
#line 2385 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367835_28_non_const_z_to_go;
#line 2386 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367836_10_non_const_x_extension;
#line 2386 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367836_23_non_const_y_extension;
#line 2387 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367837_7_non_const_voxel_x;
#line 2387 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367837_16_non_const_voxel_y;
#line 2387 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367837_25_non_const_voxel_z;
#line 2389 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367839_7_non_const_voxel;
#line 2389 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367839_14_non_const_voxel_out;
#line 2390 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367840_8_non_const_end_walk;
#line 2391 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_367841_8_non_const_debug_run;
#line 2402 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367852_10_non_const_x_replace;
#line 2415 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367865_7_non_const_voxel_x_out;
#line 2416 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367866_7_non_const_voxel_y_out;
#line 2417 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367867_7_non_const_voxel_z_out;
#line 2443 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367893_7_non_const_j;
#line 2444 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367894_7_non_const_j_low_limit;
#line 2445 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367895_7_non_const_j_high_limit;
#line 2380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_10_non_const_x_move = (0.0);
#line 2380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_22_non_const_y_move = (0.0);
#line 2380 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_34_non_const_z_move = (0.0);
#line 2391 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367841_8_non_const_debug_run = ((__nv_bool)1);
#line 2395 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367828_7_non_const_x_move_direction = (((int)((x_entry[__cuda_local_var_367820_6_non_const_i]) <= (x_exit[__cuda_local_var_367820_6_non_const_i]))) - ((int)((x_entry[__cuda_local_var_367820_6_non_const_i]) >= (x_exit[__cuda_local_var_367820_6_non_const_i]))));
#line 2396 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367828_25_non_const_y_move_direction = (((int)((y_entry[__cuda_local_var_367820_6_non_const_i]) <= (y_exit[__cuda_local_var_367820_6_non_const_i]))) - ((int)((y_entry[__cuda_local_var_367820_6_non_const_i]) >= (y_exit[__cuda_local_var_367820_6_non_const_i]))));
#line 2397 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367828_43_non_const_z_move_direction = (((int)((z_entry[__cuda_local_var_367820_6_non_const_i]) <= (z_exit[__cuda_local_var_367820_6_non_const_i]))) - ((int)((z_entry[__cuda_local_var_367820_6_non_const_i]) >= (z_exit[__cuda_local_var_367820_6_non_const_i]))));
#line 2398 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go = (_Z15x_remaining_GPUdiRi(((double)(x_entry[__cuda_local_var_367820_6_non_const_i])), __cuda_local_var_367828_7_non_const_x_move_direction, (&__cuda_local_var_367837_7_non_const_voxel_x)));
#line 2399 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go = (_Z15y_remaining_GPUdiRi(((double)(y_entry[__cuda_local_var_367820_6_non_const_i])), (-__cuda_local_var_367828_25_non_const_y_move_direction), (&__cuda_local_var_367837_16_non_const_voxel_y)));
#line 2400 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_28_non_const_z_to_go = (_Z15z_remaining_GPUdiRi(((double)(z_entry[__cuda_local_var_367820_6_non_const_i])), (-__cuda_local_var_367828_43_non_const_z_move_direction), (&__cuda_local_var_367837_25_non_const_voxel_z)));
#line 2401 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367839_7_non_const_voxel = ((__cuda_local_var_367837_7_non_const_voxel_x + (__cuda_local_var_367837_16_non_const_voxel_y * 200)) + ((__cuda_local_var_367837_25_non_const_voxel_z * 200) * 200));
#line 2402 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367852_10_non_const_x_replace = ((double)(x_exit[__cuda_local_var_367820_6_non_const_i]));
#line 2403 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367839_14_non_const_voxel_out = (_Z20position_2_voxel_GPUddd(__cuda_local_var_367852_10_non_const_x_replace, ((double)(y_exit[__cuda_local_var_367820_6_non_const_i])), ((double)(z_exit[__cuda_local_var_367820_6_non_const_i]))));
#line 2409 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367829_10_non_const_delta_yx = ((double)( fdividef(((__T2148 = ((y_exit[__cuda_local_var_367820_6_non_const_i]) - (y_entry[__cuda_local_var_367820_6_non_const_i]))) , (fabsf(__T2148))) , ((__T2149 = ((x_exit[__cuda_local_var_367820_6_non_const_i]) - (x_entry[__cuda_local_var_367820_6_non_const_i]))) , (fabsf(__T2149))))));
#line 2410 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367829_20_non_const_delta_zx = ((double)( fdividef(((__T2150 = ((z_exit[__cuda_local_var_367820_6_non_const_i]) - (z_entry[__cuda_local_var_367820_6_non_const_i]))) , (fabsf(__T2150))) , ((__T2151 = ((x_exit[__cuda_local_var_367820_6_non_const_i]) - (x_entry[__cuda_local_var_367820_6_non_const_i]))) , (fabsf(__T2151))))));
#line 2411 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367829_30_non_const_delta_zy = ((double)( fdividef(((__T2152 = ((z_exit[__cuda_local_var_367820_6_non_const_i]) - (z_entry[__cuda_local_var_367820_6_non_const_i]))) , (fabsf(__T2152))) , ((__T2153 = ((y_exit[__cuda_local_var_367820_6_non_const_i]) - (y_entry[__cuda_local_var_367820_6_non_const_i]))) , (fabsf(__T2153))))));
#line 2415 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367865_7_non_const_voxel_x_out = ((int)__double2int_rz((double)(( fdivide((((double)(x_exit[__cuda_local_var_367820_6_non_const_i])) + (8.0)) , (0.080000000000000002))))));
#line 2416 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367866_7_non_const_voxel_y_out = ((int)__double2int_rz((double)(( fdivide(((8.0) - ((double)(y_exit[__cuda_local_var_367820_6_non_const_i]))) , (0.080000000000000002))))));
#line 2417 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367867_7_non_const_voxel_z_out = ((int)__double2int_rz((double)(( fdivide(((3.0) - ((double)(z_exit[__cuda_local_var_367820_6_non_const_i]))) , (0.25))))));
#line 2418 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367841_8_non_const_debug_run)
#line 2419 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2420 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"bin_num[i] = %d NUM_BINS = %d"), (bin_num[__cuda_local_var_367820_6_non_const_i]), 756000);
#line 2421 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_move_direction = %d y_move_direction = %d z_move_direction = %d\n"), __cuda_local_var_367828_7_non_const_x_move_direction, __cuda_local_var_367828_25_non_const_y_move_direction, __cuda_local_var_367828_43_non_const_z_move_direction);
#line 2422 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"VOXEL_WIDTH = %3f VOXEL_HEIGHT = %3f SLICE_THICKNESS = %3f\n"), (0.080000000000000002), (0.080000000000000002), (0.25));
#line 2423 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"voxel_x_in = %d voxel_y_in = %d voxel_z_in = %d\n"), __cuda_local_var_367837_7_non_const_voxel_x, __cuda_local_var_367837_16_non_const_voxel_y, __cuda_local_var_367837_25_non_const_voxel_z);
#line 2424 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"voxel_x_out = %d voxel_y_out = %d voxel_z_out = %d\n"), __cuda_local_var_367865_7_non_const_voxel_x_out, __cuda_local_var_367866_7_non_const_voxel_y_out, __cuda_local_var_367867_7_non_const_voxel_z_out);
#line 2425 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"voxel_in = %d voxel_out = %d\n"), __cuda_local_var_367839_7_non_const_voxel, __cuda_local_var_367839_14_non_const_voxel_out);
#line 2426 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_entry[i] = %3f y_entry[i] = %3f z_entry[i] = %3f\n"), ((double)(x_entry[__cuda_local_var_367820_6_non_const_i])), ((double)(y_entry[__cuda_local_var_367820_6_non_const_i])), ((double)(z_entry[__cuda_local_var_367820_6_non_const_i])));
#line 2427 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_exit[i] = %8e y_exit[i] = %3f z_exit[i] = %3f\n"), __cuda_local_var_367852_10_non_const_x_replace, ((double)(y_exit[__cuda_local_var_367820_6_non_const_i])), ((double)(z_exit[__cuda_local_var_367820_6_non_const_i])));
#line 2428 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_replace > -7.28\?: %d\n"), ((int)(__cuda_local_var_367852_10_non_const_x_replace > (-7.2800000000000002))));
#line 2429 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_replace < -7.28\?: %d\n"), ((int)(__cuda_local_var_367852_10_non_const_x_replace < (-7.2800000000000002))));
#line 2430 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_replace = -7.28\?: %d\n"), ((int)(__cuda_local_var_367852_10_non_const_x_replace == (-7.2800000000000002))));
#line 2431 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_exit[i] > -7.28\?: %d\n"), ((int)(((double)(x_exit[__cuda_local_var_367820_6_non_const_i])) > (-7.2800000000000002))));
#line 2432 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_exit[i] < -7.28\?: %d\n"), ((int)(((double)(x_exit[__cuda_local_var_367820_6_non_const_i])) < (-7.2800000000000002))));
#line 2433 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_exit[i] = -7.28\?: %d\n"), ((int)(((double)(x_exit[__cuda_local_var_367820_6_non_const_i])) == (-7.2800000000000002))));
#line 2434 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_exit[i] voxel %8f\n"), ( fdivide((((double)(x_exit[__cuda_local_var_367820_6_non_const_i])) + (8.0)) , (0.080000000000000002))));
#line 2436 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"delta_yx = %3f delta_zx = %3f delta_zy = %3f\n"), __cuda_local_var_367829_10_non_const_delta_yx, __cuda_local_var_367829_20_non_const_delta_zx, __cuda_local_var_367829_30_non_const_delta_zy);
#line 2437 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_to_go = %3f y_to_go = %3f z_to_go = %3f\n"), __cuda_local_var_367835_10_non_const_x_to_go, __cuda_local_var_367835_19_non_const_y_to_go, __cuda_local_var_367835_28_non_const_z_to_go);
#line 2438 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2439 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367840_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_367839_7_non_const_voxel == __cuda_local_var_367839_14_non_const_voxel_out) || (__cuda_local_var_367837_7_non_const_voxel_x >= 200)) || (__cuda_local_var_367837_16_non_const_voxel_y >= 200)) || (__cuda_local_var_367837_25_non_const_voxel_z >= 24)));
#line 2440 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_367840_8_non_const_end_walk)) {
#line 2441 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SC_hull[__cuda_local_var_367839_7_non_const_voxel]) = ((__nv_bool)0); }
#line 2443 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367893_7_non_const_j = 0;
#line 2444 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367894_7_non_const_j_low_limit = 185;
#line 2445 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367895_7_non_const_j_high_limit = 250;
#line 2449 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367828_43_non_const_z_move_direction != 0)
#line 2450 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2451 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit)) {
#line 2452 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"z_exit[i] != z_entry[i]\n")); }
#line 2453 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_367840_8_non_const_end_walk))
#line 2454 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2456 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367836_10_non_const_x_extension = (__cuda_local_var_367829_20_non_const_delta_zx * __cuda_local_var_367835_10_non_const_x_to_go);
#line 2457 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367836_23_non_const_y_extension = (__cuda_local_var_367829_30_non_const_delta_zy * __cuda_local_var_367835_19_non_const_y_to_go);
#line 2458 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_367835_28_non_const_z_to_go <= __cuda_local_var_367836_10_non_const_x_extension) && (__cuda_local_var_367835_28_non_const_z_to_go <= __cuda_local_var_367836_23_non_const_y_extension))
#line 2459 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2461 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_10_non_const_x_move = ( fdivide(__cuda_local_var_367835_28_non_const_z_to_go , __cuda_local_var_367829_20_non_const_delta_zx));
#line 2462 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_22_non_const_y_move = ( fdivide(__cuda_local_var_367835_28_non_const_z_to_go , __cuda_local_var_367829_30_non_const_delta_zy));
#line 2463 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_34_non_const_z_move = __cuda_local_var_367835_28_non_const_z_to_go;
#line 2464 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go -= (( fdivide(__cuda_local_var_367835_28_non_const_z_to_go , __cuda_local_var_367829_20_non_const_delta_zx)) * ((double)(abs(__cuda_local_var_367828_7_non_const_x_move_direction))));
#line 2465 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go -= (( fdivide(__cuda_local_var_367835_28_non_const_z_to_go , __cuda_local_var_367829_30_non_const_delta_zy)) * ((double)(abs(__cuda_local_var_367828_25_non_const_y_move_direction))));
#line 2466 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_28_non_const_z_to_go = (0.25);
#line 2467 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_25_non_const_voxel_z -= __cuda_local_var_367828_43_non_const_z_move_direction;
#line 2468 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367835_10_non_const_x_to_go <= (9.9999999999999995e-008))
#line 2469 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2470 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_7_non_const_voxel_x += __cuda_local_var_367828_7_non_const_x_move_direction;
#line 2471 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go = (0.080000000000000002);
#line 2472 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2473 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367835_19_non_const_y_to_go <= (9.9999999999999995e-008))
#line 2474 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2475 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_16_non_const_voxel_y -= __cuda_local_var_367828_25_non_const_y_move_direction;
#line 2476 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go = (0.080000000000000002);
#line 2477 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2478 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit))
#line 2479 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2480 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"z_to_go <= x_extension && z_to_go <= y_extension\n"));
#line 2481 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_move = %3f y_move = %3f z_move = %3f\n"), __cuda_local_var_367830_10_non_const_x_move, __cuda_local_var_367830_22_non_const_y_move, __cuda_local_var_367830_34_non_const_z_move);
#line 2482 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_to_go = %8f y_to_go = %3f z_to_go = %3f\n"), __cuda_local_var_367835_10_non_const_x_to_go, __cuda_local_var_367835_19_non_const_y_to_go, __cuda_local_var_367835_28_non_const_z_to_go);
#line 2483 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2484 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} else  {
#line 2486 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367836_10_non_const_x_extension <= __cuda_local_var_367836_23_non_const_y_extension)
#line 2487 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2489 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_10_non_const_x_move = __cuda_local_var_367835_10_non_const_x_to_go;
#line 2490 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_22_non_const_y_move = (__cuda_local_var_367829_10_non_const_delta_yx * __cuda_local_var_367835_10_non_const_x_to_go);
#line 2491 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_34_non_const_z_move = (__cuda_local_var_367829_20_non_const_delta_zx * __cuda_local_var_367835_10_non_const_x_to_go);
#line 2492 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go -= ((__cuda_local_var_367829_10_non_const_delta_yx * __cuda_local_var_367835_10_non_const_x_to_go) * ((double)(abs(__cuda_local_var_367828_25_non_const_y_move_direction))));
#line 2493 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_28_non_const_z_to_go -= ((__cuda_local_var_367829_20_non_const_delta_zx * __cuda_local_var_367835_10_non_const_x_to_go) * ((double)(abs(__cuda_local_var_367828_43_non_const_z_move_direction))));
#line 2494 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go = (0.080000000000000002);
#line 2495 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_7_non_const_voxel_x += __cuda_local_var_367828_7_non_const_x_move_direction;
#line 2496 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367835_19_non_const_y_to_go <= (9.9999999999999995e-008))
#line 2497 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2498 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go = (0.080000000000000002);
#line 2499 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_16_non_const_voxel_y -= __cuda_local_var_367828_25_non_const_y_move_direction;
#line 2500 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2501 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit))
#line 2502 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2503 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)" x_extension <= y_extension \n"));
#line 2504 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_move = %3f y_move = %3f z_move = %3f\n"), __cuda_local_var_367830_10_non_const_x_move, __cuda_local_var_367830_22_non_const_y_move, __cuda_local_var_367830_34_non_const_z_move);
#line 2505 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_to_go = %3f y_to_go = %3f z_to_go = %3f\n"), __cuda_local_var_367835_10_non_const_x_to_go, __cuda_local_var_367835_19_non_const_y_to_go, __cuda_local_var_367835_28_non_const_z_to_go);
#line 2506 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2507 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2510 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2510 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2512 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_10_non_const_x_move = ( fdivide(__cuda_local_var_367835_19_non_const_y_to_go , __cuda_local_var_367829_10_non_const_delta_yx));
#line 2513 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_22_non_const_y_move = __cuda_local_var_367835_19_non_const_y_to_go;
#line 2514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_34_non_const_z_move = (__cuda_local_var_367829_30_non_const_delta_zy * __cuda_local_var_367835_19_non_const_y_to_go);
#line 2515 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go -= (( fdivide(__cuda_local_var_367835_19_non_const_y_to_go , __cuda_local_var_367829_10_non_const_delta_yx)) * ((double)(abs(__cuda_local_var_367828_7_non_const_x_move_direction))));
#line 2516 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_28_non_const_z_to_go -= ((__cuda_local_var_367829_30_non_const_delta_zy * __cuda_local_var_367835_19_non_const_y_to_go) * ((double)(abs(__cuda_local_var_367828_43_non_const_z_move_direction))));
#line 2517 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go = (0.080000000000000002);
#line 2518 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_16_non_const_voxel_y -= __cuda_local_var_367828_25_non_const_y_move_direction;
#line 2519 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit))
#line 2520 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2521 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)" y_extension < x_extension \n"));
#line 2522 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_move = %3f y_move = %3f z_move = %3f\n"), __cuda_local_var_367830_10_non_const_x_move, __cuda_local_var_367830_22_non_const_y_move, __cuda_local_var_367830_34_non_const_z_move);
#line 2523 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_to_go = %3f y_to_go = %3f z_to_go = %3f\n"), __cuda_local_var_367835_10_non_const_x_to_go, __cuda_local_var_367835_19_non_const_y_to_go, __cuda_local_var_367835_28_non_const_z_to_go);
#line 2524 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2525 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2529 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_25_non_const_voxel_z = (max(__cuda_local_var_367837_25_non_const_voxel_z, 0));
#line 2530 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367839_7_non_const_voxel = ((__cuda_local_var_367837_7_non_const_voxel_x + (__cuda_local_var_367837_16_non_const_voxel_y * 200)) + ((__cuda_local_var_367837_25_non_const_voxel_z * 200) * 200));
#line 2531 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit)) {
#line 2532 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n"), __cuda_local_var_367837_7_non_const_voxel_x, __cuda_local_var_367837_16_non_const_voxel_y, __cuda_local_var_367837_25_non_const_voxel_z, __cuda_local_var_367839_7_non_const_voxel); }
#line 2533 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367840_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_367839_7_non_const_voxel == __cuda_local_var_367839_14_non_const_voxel_out) || (__cuda_local_var_367837_7_non_const_voxel_x >= 200)) || (__cuda_local_var_367837_16_non_const_voxel_y >= 200)) || (__cuda_local_var_367837_25_non_const_voxel_z >= 24)));
#line 2534 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_367840_8_non_const_end_walk)) {
#line 2535 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SC_hull[__cuda_local_var_367839_7_non_const_voxel]) = ((__nv_bool)0); }
#line 2536 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367893_7_non_const_j++;
#line 2537 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2538 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2540 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2540 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2541 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit)) {
#line 2542 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"z_exit[i] == z_entry[i]\n")); }
#line 2543 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_367840_8_non_const_end_walk))
#line 2544 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2546 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367836_23_non_const_y_extension = ( fdivide(__cuda_local_var_367835_19_non_const_y_to_go , __cuda_local_var_367829_10_non_const_delta_yx));
#line 2548 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367835_10_non_const_x_to_go <= __cuda_local_var_367836_23_non_const_y_extension)
#line 2549 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2551 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_10_non_const_x_move = __cuda_local_var_367835_10_non_const_x_to_go;
#line 2552 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_22_non_const_y_move = (__cuda_local_var_367829_10_non_const_delta_yx * __cuda_local_var_367835_10_non_const_x_to_go);
#line 2553 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go -= ((__cuda_local_var_367829_10_non_const_delta_yx * __cuda_local_var_367835_10_non_const_x_to_go) * ((double)(abs(__cuda_local_var_367828_25_non_const_y_move_direction))));
#line 2554 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go = (0.080000000000000002);
#line 2555 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_7_non_const_voxel_x += __cuda_local_var_367828_7_non_const_x_move_direction;
#line 2556 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_367835_19_non_const_y_to_go <= (9.9999999999999995e-008))
#line 2557 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2558 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go = (0.080000000000000002);
#line 2559 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_16_non_const_voxel_y -= __cuda_local_var_367828_25_non_const_y_move_direction;
#line 2560 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2561 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit))
#line 2562 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2563 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)" x_to_go <= y_extension\n"));
#line 2564 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_move = %3f y_move = %3f \n"), __cuda_local_var_367830_10_non_const_x_move, __cuda_local_var_367830_22_non_const_y_move);
#line 2565 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_to_go = %3f y_to_go = %3f\n"), __cuda_local_var_367835_10_non_const_x_to_go, __cuda_local_var_367835_19_non_const_y_to_go);
#line 2566 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2567 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2570 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2572 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_10_non_const_x_move = ( fdivide(__cuda_local_var_367835_19_non_const_y_to_go , __cuda_local_var_367829_10_non_const_delta_yx));
#line 2573 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367830_22_non_const_y_move = __cuda_local_var_367835_19_non_const_y_to_go;
#line 2574 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_10_non_const_x_to_go -= (( fdivide(__cuda_local_var_367835_19_non_const_y_to_go , __cuda_local_var_367829_10_non_const_delta_yx)) * ((double)(abs(__cuda_local_var_367828_7_non_const_x_move_direction))));
#line 2575 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367835_19_non_const_y_to_go = (0.080000000000000002);
#line 2576 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_16_non_const_voxel_y -= __cuda_local_var_367828_25_non_const_y_move_direction;
#line 2577 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit))
#line 2578 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2579 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)" y_extension < x_extension\n"));
#line 2580 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_move = %3f y_move = %3f \n"), __cuda_local_var_367830_10_non_const_x_move, __cuda_local_var_367830_22_non_const_y_move);
#line 2581 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"x_to_go = %3f y_to_go = %3f\n"), __cuda_local_var_367835_10_non_const_x_to_go, __cuda_local_var_367835_19_non_const_y_to_go);
#line 2582 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2583 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2586 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367837_25_non_const_voxel_z = (max(__cuda_local_var_367837_25_non_const_voxel_z, 0));
#line 2587 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367839_7_non_const_voxel = ((__cuda_local_var_367837_7_non_const_voxel_x + (__cuda_local_var_367837_16_non_const_voxel_y * 200)) + ((__cuda_local_var_367837_25_non_const_voxel_z * 200) * 200));
#line 2588 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_367841_8_non_const_debug_run) && (__cuda_local_var_367893_7_non_const_j < __cuda_local_var_367895_7_non_const_j_high_limit)) && (__cuda_local_var_367893_7_non_const_j > __cuda_local_var_367894_7_non_const_j_low_limit)) {
#line 2589 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"voxel_x_in = %d voxel_y_in = %d voxel_z_in = %d\n"), __cuda_local_var_367837_7_non_const_voxel_x, __cuda_local_var_367837_16_non_const_voxel_y, __cuda_local_var_367837_25_non_const_voxel_z); }
#line 2590 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367840_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_367839_7_non_const_voxel == __cuda_local_var_367839_14_non_const_voxel_out) || (__cuda_local_var_367837_7_non_const_voxel_x >= 200)) || (__cuda_local_var_367837_16_non_const_voxel_y >= 200)) || (__cuda_local_var_367837_25_non_const_voxel_z >= 24)));
#line 2591 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_367840_8_non_const_end_walk)) {
#line 2592 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SC_hull[__cuda_local_var_367839_7_non_const_voxel]) = ((__nv_bool)0); }
#line 2593 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367893_7_non_const_j++;
#line 2594 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2596 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2597 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 2598 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2610 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z7MSC_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_(
#line 2612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
const int num_histories, 
#line 2612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *MSC_counts, 
#line 2612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_num, 
#line 2612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *missed_recon_volume, 
#line 2612 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *WEPL, 
#line 2613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_entry, 
#line 2613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_entry, 
#line 2613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_entry, 
#line 2613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_exit, 
#line 2613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_exit, 
#line 2613 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_exit){
#line 2615 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2616 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368066_6_non_const_i;
#line 2616 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368066_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 2618 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_368066_6_non_const_i < num_histories) && (!(missed_recon_volume[__cuda_local_var_368066_6_non_const_i]))) && (((double)(WEPL[__cuda_local_var_368066_6_non_const_i])) <= (0.0)))
#line 2619 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T2154;
 float __T2155;
 float __T2156;
 float __T2157;
 float __T2158;
 float __T2159;
#line 2623 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368073_7_non_const_x_move_direction;
#line 2623 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368073_25_non_const_y_move_direction;
#line 2623 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368073_43_non_const_z_move_direction;
#line 2624 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368074_10_non_const_delta_yx;
#line 2624 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368074_20_non_const_delta_zx;
#line 2624 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368074_30_non_const_delta_zy;
#line 2625 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368075_10_non_const_x_move;
#line 2625 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368075_22_non_const_y_move;
#line 2625 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368075_34_non_const_z_move;
#line 2630 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368080_10_non_const_x_to_go;
#line 2630 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368080_19_non_const_y_to_go;
#line 2630 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368080_28_non_const_z_to_go;
#line 2631 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368081_10_non_const_x_extension;
#line 2631 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368081_23_non_const_y_extension;
#line 2632 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368082_7_non_const_voxel_x;
#line 2632 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368082_16_non_const_voxel_y;
#line 2632 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368082_25_non_const_voxel_z;
#line 2634 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368084_7_non_const_voxel;
#line 2634 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368084_14_non_const_voxel_out;
#line 2635 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_368085_8_non_const_end_walk;
#line 2625 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_10_non_const_x_move = (0.0);
#line 2625 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_22_non_const_y_move = (0.0);
#line 2625 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_34_non_const_z_move = (0.0);
#line 2639 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368073_7_non_const_x_move_direction = (((int)((x_entry[__cuda_local_var_368066_6_non_const_i]) <= (x_exit[__cuda_local_var_368066_6_non_const_i]))) - ((int)((x_entry[__cuda_local_var_368066_6_non_const_i]) >= (x_exit[__cuda_local_var_368066_6_non_const_i]))));
#line 2640 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368073_25_non_const_y_move_direction = (((int)((y_entry[__cuda_local_var_368066_6_non_const_i]) <= (y_exit[__cuda_local_var_368066_6_non_const_i]))) - ((int)((y_entry[__cuda_local_var_368066_6_non_const_i]) >= (y_exit[__cuda_local_var_368066_6_non_const_i]))));
#line 2641 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368073_43_non_const_z_move_direction = (((int)((z_entry[__cuda_local_var_368066_6_non_const_i]) <= (z_exit[__cuda_local_var_368066_6_non_const_i]))) - ((int)((z_entry[__cuda_local_var_368066_6_non_const_i]) >= (z_exit[__cuda_local_var_368066_6_non_const_i]))));
#line 2642 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go = (_Z15x_remaining_GPUdiRi(((double)(x_entry[__cuda_local_var_368066_6_non_const_i])), __cuda_local_var_368073_7_non_const_x_move_direction, (&__cuda_local_var_368082_7_non_const_voxel_x)));
#line 2643 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go = (_Z15y_remaining_GPUdiRi(((double)(y_entry[__cuda_local_var_368066_6_non_const_i])), (-__cuda_local_var_368073_25_non_const_y_move_direction), (&__cuda_local_var_368082_16_non_const_voxel_y)));
#line 2644 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_28_non_const_z_to_go = (_Z15z_remaining_GPUdiRi(((double)(z_entry[__cuda_local_var_368066_6_non_const_i])), (-__cuda_local_var_368073_43_non_const_z_move_direction), (&__cuda_local_var_368082_25_non_const_voxel_z)));
#line 2645 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368084_7_non_const_voxel = ((__cuda_local_var_368082_7_non_const_voxel_x + (__cuda_local_var_368082_16_non_const_voxel_y * 200)) + ((__cuda_local_var_368082_25_non_const_voxel_z * 200) * 200));
#line 2646 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368084_14_non_const_voxel_out = (_Z20position_2_voxel_GPUddd(((double)(x_exit[__cuda_local_var_368066_6_non_const_i])), ((double)(y_exit[__cuda_local_var_368066_6_non_const_i])), ((double)(z_exit[__cuda_local_var_368066_6_non_const_i]))));
#line 2651 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368074_10_non_const_delta_yx = ((double)( fdividef(((__T2154 = ((y_exit[__cuda_local_var_368066_6_non_const_i]) - (y_entry[__cuda_local_var_368066_6_non_const_i]))) , (fabsf(__T2154))) , ((__T2155 = ((x_exit[__cuda_local_var_368066_6_non_const_i]) - (x_entry[__cuda_local_var_368066_6_non_const_i]))) , (fabsf(__T2155))))));
#line 2652 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368074_20_non_const_delta_zx = ((double)( fdividef(((__T2156 = ((z_exit[__cuda_local_var_368066_6_non_const_i]) - (z_entry[__cuda_local_var_368066_6_non_const_i]))) , (fabsf(__T2156))) , ((__T2157 = ((x_exit[__cuda_local_var_368066_6_non_const_i]) - (x_entry[__cuda_local_var_368066_6_non_const_i]))) , (fabsf(__T2157))))));
#line 2653 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368074_30_non_const_delta_zy = ((double)( fdividef(((__T2158 = ((z_exit[__cuda_local_var_368066_6_non_const_i]) - (z_entry[__cuda_local_var_368066_6_non_const_i]))) , (fabsf(__T2158))) , ((__T2159 = ((y_exit[__cuda_local_var_368066_6_non_const_i]) - (y_entry[__cuda_local_var_368066_6_non_const_i]))) , (fabsf(__T2159))))));
#line 2657 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368085_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_368084_7_non_const_voxel == __cuda_local_var_368084_14_non_const_voxel_out) || (__cuda_local_var_368082_7_non_const_voxel_x >= 200)) || (__cuda_local_var_368082_16_non_const_voxel_y >= 200)) || (__cuda_local_var_368082_25_non_const_voxel_z >= 24)));
#line 2658 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_368085_8_non_const_end_walk)) {
#line 2659 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((MSC_counts + __cuda_local_var_368084_7_non_const_voxel), 1); }
#line 2663 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368073_43_non_const_z_move_direction != 0)
#line 2664 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2666 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_368085_8_non_const_end_walk))
#line 2667 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2669 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368081_10_non_const_x_extension = (__cuda_local_var_368074_20_non_const_delta_zx * __cuda_local_var_368080_10_non_const_x_to_go);
#line 2670 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368081_23_non_const_y_extension = (__cuda_local_var_368074_30_non_const_delta_zy * __cuda_local_var_368080_19_non_const_y_to_go);
#line 2671 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_368080_28_non_const_z_to_go <= __cuda_local_var_368081_10_non_const_x_extension) && (__cuda_local_var_368080_28_non_const_z_to_go <= __cuda_local_var_368081_23_non_const_y_extension))
#line 2672 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2674 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_10_non_const_x_move = ( fdivide(__cuda_local_var_368080_28_non_const_z_to_go , __cuda_local_var_368074_20_non_const_delta_zx));
#line 2675 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_22_non_const_y_move = ( fdivide(__cuda_local_var_368080_28_non_const_z_to_go , __cuda_local_var_368074_30_non_const_delta_zy));
#line 2676 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_34_non_const_z_move = __cuda_local_var_368080_28_non_const_z_to_go;
#line 2677 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go -= (__cuda_local_var_368075_10_non_const_x_move * ((double)(abs(__cuda_local_var_368073_7_non_const_x_move_direction))));
#line 2678 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go -= (__cuda_local_var_368075_22_non_const_y_move * ((double)(abs(__cuda_local_var_368073_25_non_const_y_move_direction))));
#line 2681 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_28_non_const_z_to_go = (0.25);
#line 2682 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_25_non_const_voxel_z -= __cuda_local_var_368073_43_non_const_z_move_direction;
#line 2683 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368080_10_non_const_x_to_go == (0.0))
#line 2684 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2685 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_7_non_const_voxel_x += __cuda_local_var_368073_7_non_const_x_move_direction;
#line 2686 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go = (0.080000000000000002);
#line 2687 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2688 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368080_19_non_const_y_to_go == (0.0))
#line 2689 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2690 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_16_non_const_voxel_y -= __cuda_local_var_368073_25_non_const_y_move_direction;
#line 2691 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go = (0.080000000000000002);
#line 2692 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2693 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} else  {
#line 2695 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368081_10_non_const_x_extension <= __cuda_local_var_368081_23_non_const_y_extension)
#line 2696 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2698 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_10_non_const_x_move = __cuda_local_var_368080_10_non_const_x_to_go;
#line 2699 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_22_non_const_y_move = (__cuda_local_var_368074_10_non_const_delta_yx * __cuda_local_var_368080_10_non_const_x_to_go);
#line 2700 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_34_non_const_z_move = (__cuda_local_var_368074_20_non_const_delta_zx * __cuda_local_var_368080_10_non_const_x_to_go);
#line 2701 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go = (0.080000000000000002);
#line 2702 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go -= (__cuda_local_var_368075_22_non_const_y_move * ((double)(abs(__cuda_local_var_368073_25_non_const_y_move_direction))));
#line 2703 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_28_non_const_z_to_go -= (__cuda_local_var_368075_34_non_const_z_move * ((double)(abs(__cuda_local_var_368073_43_non_const_z_move_direction))));
#line 2707 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_7_non_const_voxel_x += __cuda_local_var_368073_7_non_const_x_move_direction;
#line 2708 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368080_19_non_const_y_to_go == (0.0))
#line 2709 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2710 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go = (0.080000000000000002);
#line 2711 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_16_non_const_voxel_y -= __cuda_local_var_368073_25_non_const_y_move_direction;
#line 2712 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2713 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2716 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2716 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2718 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_10_non_const_x_move = ( fdivide(__cuda_local_var_368080_19_non_const_y_to_go , __cuda_local_var_368074_10_non_const_delta_yx));
#line 2719 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_22_non_const_y_move = __cuda_local_var_368080_19_non_const_y_to_go;
#line 2720 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_34_non_const_z_move = (__cuda_local_var_368074_30_non_const_delta_zy * __cuda_local_var_368080_19_non_const_y_to_go);
#line 2721 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go -= (__cuda_local_var_368075_10_non_const_x_move * ((double)(abs(__cuda_local_var_368073_7_non_const_x_move_direction))));
#line 2722 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go = (0.080000000000000002);
#line 2723 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_28_non_const_z_to_go -= (__cuda_local_var_368075_34_non_const_z_move * ((double)(abs(__cuda_local_var_368073_43_non_const_z_move_direction))));
#line 2727 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_16_non_const_voxel_y -= __cuda_local_var_368073_25_non_const_y_move_direction;
#line 2728 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2732 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368084_7_non_const_voxel = ((__cuda_local_var_368082_7_non_const_voxel_x + (__cuda_local_var_368082_16_non_const_voxel_y * 200)) + ((__cuda_local_var_368082_25_non_const_voxel_z * 200) * 200));
#line 2733 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368085_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_368084_7_non_const_voxel == __cuda_local_var_368084_14_non_const_voxel_out) || (__cuda_local_var_368082_7_non_const_voxel_x >= 200)) || (__cuda_local_var_368082_16_non_const_voxel_y >= 200)) || (__cuda_local_var_368082_25_non_const_voxel_z >= 24)));
#line 2734 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_368085_8_non_const_end_walk)) {
#line 2735 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((MSC_counts + __cuda_local_var_368084_7_non_const_voxel), 1); }
#line 2736 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2737 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2739 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2739 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2741 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_368085_8_non_const_end_walk))
#line 2742 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2744 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368081_23_non_const_y_extension = ( fdivide(__cuda_local_var_368080_19_non_const_y_to_go , __cuda_local_var_368074_10_non_const_delta_yx));
#line 2746 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368080_10_non_const_x_to_go <= __cuda_local_var_368081_23_non_const_y_extension)
#line 2747 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2749 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_10_non_const_x_move = __cuda_local_var_368080_10_non_const_x_to_go;
#line 2750 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_22_non_const_y_move = (__cuda_local_var_368074_10_non_const_delta_yx * __cuda_local_var_368080_10_non_const_x_to_go);
#line 2751 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go = (0.080000000000000002);
#line 2753 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go -= (__cuda_local_var_368075_22_non_const_y_move * ((double)(abs(__cuda_local_var_368073_25_non_const_y_move_direction))));
#line 2754 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_7_non_const_voxel_x += __cuda_local_var_368073_7_non_const_x_move_direction;
#line 2755 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368080_19_non_const_y_to_go == (0.0))
#line 2756 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2757 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go = (0.080000000000000002);
#line 2758 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_16_non_const_voxel_y -= __cuda_local_var_368073_25_non_const_y_move_direction;
#line 2759 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2760 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2763 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2763 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2765 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_10_non_const_x_move = ( fdivide(__cuda_local_var_368080_19_non_const_y_to_go , __cuda_local_var_368074_10_non_const_delta_yx));
#line 2766 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368075_22_non_const_y_move = __cuda_local_var_368080_19_non_const_y_to_go;
#line 2767 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_10_non_const_x_to_go -= (__cuda_local_var_368075_10_non_const_x_move * ((double)(abs(__cuda_local_var_368073_7_non_const_x_move_direction))));
#line 2769 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368080_19_non_const_y_to_go = (0.080000000000000002);
#line 2770 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368082_16_non_const_voxel_y -= __cuda_local_var_368073_25_non_const_y_move_direction;
#line 2771 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2774 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368084_7_non_const_voxel = ((__cuda_local_var_368082_7_non_const_voxel_x + (__cuda_local_var_368082_16_non_const_voxel_y * 200)) + ((__cuda_local_var_368082_25_non_const_voxel_z * 200) * 200));
#line 2775 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368085_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_368084_7_non_const_voxel == __cuda_local_var_368084_14_non_const_voxel_out) || (__cuda_local_var_368082_7_non_const_voxel_x >= 200)) || (__cuda_local_var_368082_16_non_const_voxel_y >= 200)) || (__cuda_local_var_368082_25_non_const_voxel_z >= 24)));
#line 2776 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_368085_8_non_const_end_walk)) {
#line 2777 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((MSC_counts + __cuda_local_var_368084_7_non_const_voxel), 1); }
#line 2778 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2780 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2781 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 2782 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2793 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z22MSC_edge_detection_GPUPi(
#line 2793 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *MSC_counts){
#line 2794 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2160;
#line 2795 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368245_6_non_const_row;
#line 2795 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368245_24_non_const_column;
#line 2795 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368245_45_non_const_slice;
#line 2796 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368246_6_non_const_voxel;
#line 2797 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_368247_8_non_const_x;
#line 2798 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_368248_8_non_const_y;
#line 2799 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368249_6_non_const_difference;
#line 2799 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368249_18_non_const_max_difference;
#line 2795 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368245_6_non_const_row = ((int)(blockIdx.y));
#line 2795 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368245_24_non_const_column = ((int)(blockIdx.x));
#line 2795 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368245_45_non_const_slice = ((int)(threadIdx.x));
#line 2796 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368246_6_non_const_voxel = ((__cuda_local_var_368245_24_non_const_column + (__cuda_local_var_368245_6_non_const_row * 200)) + ((__cuda_local_var_368245_45_non_const_slice * 200) * 200));
#line 2797 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368247_8_non_const_x = ((float)((((double)(__cuda_local_var_368245_24_non_const_column - 100)) + (0.5)) * (0.080000000000000002)));
#line 2798 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368248_8_non_const_y = ((float)((((double)(100 - __cuda_local_var_368245_6_non_const_row)) - (0.5)) * (0.080000000000000002)));
#line 2799 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368249_18_non_const_max_difference = 0;
#line 2800 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((__cuda_local_var_368245_6_non_const_row != 0) && (__cuda_local_var_368245_6_non_const_row != 199)) && (__cuda_local_var_368245_24_non_const_column != 0)) && (__cuda_local_var_368245_24_non_const_column != 199))
#line 2801 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{ {
#line 2802 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int current_row;
#line 2802 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
current_row = (__cuda_local_var_368245_6_non_const_row - 1);
#line 2802 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (current_row <= (__cuda_local_var_368245_6_non_const_row + 1)); current_row++)
#line 2803 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{ {
#line 2804 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int current_column;
#line 2804 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
current_column = (__cuda_local_var_368245_24_non_const_column - 1);
#line 2804 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (current_column <= (__cuda_local_var_368245_24_non_const_column + 1)); current_column++)
#line 2805 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2806 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368249_6_non_const_difference = ((MSC_counts[__cuda_local_var_368246_6_non_const_voxel]) - (MSC_counts[((current_column + (current_row * 200)) + ((__cuda_local_var_368245_45_non_const_slice * 200) * 200))]));
#line 2807 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368249_6_non_const_difference > __cuda_local_var_368249_18_non_const_max_difference) {
#line 2808 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368249_18_non_const_max_difference = __cuda_local_var_368249_6_non_const_difference; }
#line 2809 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2810 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2811 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2812 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"
__syncthreads();
#line 2812 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2813 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368249_18_non_const_max_difference > 50) {
#line 2814 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(MSC_counts[__cuda_local_var_368246_6_non_const_voxel]) = 0; } else  {
#line 2816 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(MSC_counts[__cuda_local_var_368246_6_non_const_voxel]) = 1; }
#line 2817 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)((_Z8_Pow_intIfET_S0_i(__cuda_local_var_368247_8_non_const_x, 2)) + (_Z8_Pow_intIfET_S0_i(__cuda_local_var_368248_8_non_const_y, 2)))) >= ((__T2160 = ((8.0) - ( fdivide((fmax((0.080000000000000002), (0.080000000000000002))) , (2.0))))) , (_Z8_Pow_intIdET_S0_i(__T2160, 2)))) {
#line 2818 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(MSC_counts[__cuda_local_var_368246_6_non_const_voxel]) = 0; } 
#line 2820 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2832 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z6SM_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_(
#line 2834 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
const int num_histories, 
#line 2834 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *SM_counts, 
#line 2834 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *bin_num, 
#line 2834 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *missed_recon_volume, 
#line 2834 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *WEPL, 
#line 2835 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_entry, 
#line 2835 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_entry, 
#line 2835 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_entry, 
#line 2835 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *x_exit, 
#line 2835 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *y_exit, 
#line 2835 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float *z_exit){
#line 2837 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2838 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368288_6_non_const_i;
#line 2838 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368288_6_non_const_i = ((int)((threadIdx.x) + ((blockIdx.x) * 1024U)));
#line 2839 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((__cuda_local_var_368288_6_non_const_i < num_histories) && (!(missed_recon_volume[__cuda_local_var_368288_6_non_const_i]))) && (((double)(WEPL[__cuda_local_var_368288_6_non_const_i])) >= (6.0)))
#line 2840 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  float __T2161;
 float __T2162;
 float __T2163;
 float __T2164;
 float __T2165;
 float __T2166;
#line 2844 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368294_7_non_const_x_move_direction;
#line 2844 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368294_25_non_const_y_move_direction;
#line 2844 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368294_43_non_const_z_move_direction;
#line 2845 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368295_10_non_const_delta_yx;
#line 2845 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368295_20_non_const_delta_zx;
#line 2845 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368295_30_non_const_delta_zy;
#line 2846 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368296_10_non_const_x_move;
#line 2846 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368296_22_non_const_y_move;
#line 2846 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368296_34_non_const_z_move;
#line 2851 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368301_10_non_const_x_to_go;
#line 2851 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368301_19_non_const_y_to_go;
#line 2851 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368301_28_non_const_z_to_go;
#line 2852 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368302_10_non_const_x_extension;
#line 2852 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368302_23_non_const_y_extension;
#line 2853 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368303_7_non_const_voxel_x;
#line 2853 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368303_16_non_const_voxel_y;
#line 2853 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368303_25_non_const_voxel_z;
#line 2855 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368305_7_non_const_voxel;
#line 2855 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368305_14_non_const_voxel_out;
#line 2856 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_368306_8_non_const_end_walk;
#line 2846 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_10_non_const_x_move = (0.0);
#line 2846 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_22_non_const_y_move = (0.0);
#line 2846 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_34_non_const_z_move = (0.0);
#line 2860 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368294_7_non_const_x_move_direction = (((int)((x_entry[__cuda_local_var_368288_6_non_const_i]) <= (x_exit[__cuda_local_var_368288_6_non_const_i]))) - ((int)((x_entry[__cuda_local_var_368288_6_non_const_i]) >= (x_exit[__cuda_local_var_368288_6_non_const_i]))));
#line 2861 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368294_25_non_const_y_move_direction = (((int)((y_entry[__cuda_local_var_368288_6_non_const_i]) <= (y_exit[__cuda_local_var_368288_6_non_const_i]))) - ((int)((y_entry[__cuda_local_var_368288_6_non_const_i]) >= (y_exit[__cuda_local_var_368288_6_non_const_i]))));
#line 2862 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368294_43_non_const_z_move_direction = (((int)((z_entry[__cuda_local_var_368288_6_non_const_i]) <= (z_exit[__cuda_local_var_368288_6_non_const_i]))) - ((int)((z_entry[__cuda_local_var_368288_6_non_const_i]) >= (z_exit[__cuda_local_var_368288_6_non_const_i]))));
#line 2863 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go = (_Z15x_remaining_GPUdiRi(((double)(x_entry[__cuda_local_var_368288_6_non_const_i])), __cuda_local_var_368294_7_non_const_x_move_direction, (&__cuda_local_var_368303_7_non_const_voxel_x)));
#line 2864 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go = (_Z15y_remaining_GPUdiRi(((double)(y_entry[__cuda_local_var_368288_6_non_const_i])), (-__cuda_local_var_368294_25_non_const_y_move_direction), (&__cuda_local_var_368303_16_non_const_voxel_y)));
#line 2865 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_28_non_const_z_to_go = (_Z15z_remaining_GPUdiRi(((double)(z_entry[__cuda_local_var_368288_6_non_const_i])), (-__cuda_local_var_368294_43_non_const_z_move_direction), (&__cuda_local_var_368303_25_non_const_voxel_z)));
#line 2866 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368305_7_non_const_voxel = ((__cuda_local_var_368303_7_non_const_voxel_x + (__cuda_local_var_368303_16_non_const_voxel_y * 200)) + ((__cuda_local_var_368303_25_non_const_voxel_z * 200) * 200));
#line 2867 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368305_14_non_const_voxel_out = (_Z20position_2_voxel_GPUddd(((double)(x_exit[__cuda_local_var_368288_6_non_const_i])), ((double)(y_exit[__cuda_local_var_368288_6_non_const_i])), ((double)(z_exit[__cuda_local_var_368288_6_non_const_i]))));
#line 2872 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368295_10_non_const_delta_yx = ((double)( fdividef(((__T2161 = ((y_exit[__cuda_local_var_368288_6_non_const_i]) - (y_entry[__cuda_local_var_368288_6_non_const_i]))) , (fabsf(__T2161))) , ((__T2162 = ((x_exit[__cuda_local_var_368288_6_non_const_i]) - (x_entry[__cuda_local_var_368288_6_non_const_i]))) , (fabsf(__T2162))))));
#line 2873 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368295_20_non_const_delta_zx = ((double)( fdividef(((__T2163 = ((z_exit[__cuda_local_var_368288_6_non_const_i]) - (z_entry[__cuda_local_var_368288_6_non_const_i]))) , (fabsf(__T2163))) , ((__T2164 = ((x_exit[__cuda_local_var_368288_6_non_const_i]) - (x_entry[__cuda_local_var_368288_6_non_const_i]))) , (fabsf(__T2164))))));
#line 2874 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368295_30_non_const_delta_zy = ((double)( fdividef(((__T2165 = ((z_exit[__cuda_local_var_368288_6_non_const_i]) - (z_entry[__cuda_local_var_368288_6_non_const_i]))) , (fabsf(__T2165))) , ((__T2166 = ((y_exit[__cuda_local_var_368288_6_non_const_i]) - (y_entry[__cuda_local_var_368288_6_non_const_i]))) , (fabsf(__T2166))))));
#line 2878 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368306_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_368305_7_non_const_voxel == __cuda_local_var_368305_14_non_const_voxel_out) || (__cuda_local_var_368303_7_non_const_voxel_x >= 200)) || (__cuda_local_var_368303_16_non_const_voxel_y >= 200)) || (__cuda_local_var_368303_25_non_const_voxel_z >= 24)));
#line 2879 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_368306_8_non_const_end_walk)) {
#line 2880 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((SM_counts + __cuda_local_var_368305_7_non_const_voxel), 1); }
#line 2884 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368294_43_non_const_z_move_direction != 0)
#line 2885 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2887 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_368306_8_non_const_end_walk))
#line 2888 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2890 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368302_10_non_const_x_extension = (__cuda_local_var_368295_20_non_const_delta_zx * __cuda_local_var_368301_10_non_const_x_to_go);
#line 2891 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368302_23_non_const_y_extension = (__cuda_local_var_368295_30_non_const_delta_zy * __cuda_local_var_368301_19_non_const_y_to_go);
#line 2892 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((__cuda_local_var_368301_28_non_const_z_to_go <= __cuda_local_var_368302_10_non_const_x_extension) && (__cuda_local_var_368301_28_non_const_z_to_go <= __cuda_local_var_368302_23_non_const_y_extension))
#line 2893 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2895 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_10_non_const_x_move = ( fdivide(__cuda_local_var_368301_28_non_const_z_to_go , __cuda_local_var_368295_20_non_const_delta_zx));
#line 2896 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_22_non_const_y_move = ( fdivide(__cuda_local_var_368301_28_non_const_z_to_go , __cuda_local_var_368295_30_non_const_delta_zy));
#line 2897 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_34_non_const_z_move = __cuda_local_var_368301_28_non_const_z_to_go;
#line 2898 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go -= (__cuda_local_var_368296_10_non_const_x_move * ((double)(abs(__cuda_local_var_368294_7_non_const_x_move_direction))));
#line 2899 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go -= (__cuda_local_var_368296_22_non_const_y_move * ((double)(abs(__cuda_local_var_368294_25_non_const_y_move_direction))));
#line 2902 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_28_non_const_z_to_go = (0.25);
#line 2903 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_25_non_const_voxel_z -= __cuda_local_var_368294_43_non_const_z_move_direction;
#line 2904 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368301_10_non_const_x_to_go == (0.0))
#line 2905 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2906 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_7_non_const_voxel_x += __cuda_local_var_368294_7_non_const_x_move_direction;
#line 2907 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go = (0.080000000000000002);
#line 2908 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2909 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368301_19_non_const_y_to_go == (0.0))
#line 2910 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2911 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_16_non_const_voxel_y -= __cuda_local_var_368294_25_non_const_y_move_direction;
#line 2912 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go = (0.080000000000000002);
#line 2913 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2914 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} else  {
#line 2916 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368302_10_non_const_x_extension <= __cuda_local_var_368302_23_non_const_y_extension)
#line 2917 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2919 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_10_non_const_x_move = __cuda_local_var_368301_10_non_const_x_to_go;
#line 2920 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_22_non_const_y_move = (__cuda_local_var_368295_10_non_const_delta_yx * __cuda_local_var_368301_10_non_const_x_to_go);
#line 2921 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_34_non_const_z_move = (__cuda_local_var_368295_20_non_const_delta_zx * __cuda_local_var_368301_10_non_const_x_to_go);
#line 2922 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go = (0.080000000000000002);
#line 2923 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go -= (__cuda_local_var_368296_22_non_const_y_move * ((double)(abs(__cuda_local_var_368294_25_non_const_y_move_direction))));
#line 2924 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_28_non_const_z_to_go -= (__cuda_local_var_368296_34_non_const_z_move * ((double)(abs(__cuda_local_var_368294_43_non_const_z_move_direction))));
#line 2928 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_7_non_const_voxel_x += __cuda_local_var_368294_7_non_const_x_move_direction;
#line 2929 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368301_19_non_const_y_to_go == (0.0))
#line 2930 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2931 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go = (0.080000000000000002);
#line 2932 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_16_non_const_voxel_y -= __cuda_local_var_368294_25_non_const_y_move_direction;
#line 2933 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2934 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2937 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2937 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2939 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_10_non_const_x_move = ( fdivide(__cuda_local_var_368301_19_non_const_y_to_go , __cuda_local_var_368295_10_non_const_delta_yx));
#line 2940 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_22_non_const_y_move = __cuda_local_var_368301_19_non_const_y_to_go;
#line 2941 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_34_non_const_z_move = (__cuda_local_var_368295_30_non_const_delta_zy * __cuda_local_var_368301_19_non_const_y_to_go);
#line 2942 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go -= (__cuda_local_var_368296_10_non_const_x_move * ((double)(abs(__cuda_local_var_368294_7_non_const_x_move_direction))));
#line 2943 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go = (0.080000000000000002);
#line 2944 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_28_non_const_z_to_go -= (__cuda_local_var_368296_34_non_const_z_move * ((double)(abs(__cuda_local_var_368294_43_non_const_z_move_direction))));
#line 2948 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_16_non_const_voxel_y -= __cuda_local_var_368294_25_non_const_y_move_direction;
#line 2949 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 2953 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368305_7_non_const_voxel = ((__cuda_local_var_368303_7_non_const_voxel_x + (__cuda_local_var_368303_16_non_const_voxel_y * 200)) + ((__cuda_local_var_368303_25_non_const_voxel_z * 200) * 200));
#line 2954 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368306_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_368305_7_non_const_voxel == __cuda_local_var_368305_14_non_const_voxel_out) || (__cuda_local_var_368303_7_non_const_voxel_x >= 200)) || (__cuda_local_var_368303_16_non_const_voxel_y >= 200)) || (__cuda_local_var_368303_25_non_const_voxel_z >= 24)));
#line 2955 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_368306_8_non_const_end_walk)) {
#line 2956 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((SM_counts + __cuda_local_var_368305_7_non_const_voxel), 1); }
#line 2957 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2958 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2960 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2960 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2962 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
while (!(__cuda_local_var_368306_8_non_const_end_walk))
#line 2963 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2965 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368302_23_non_const_y_extension = ( fdivide(__cuda_local_var_368301_19_non_const_y_to_go , __cuda_local_var_368295_10_non_const_delta_yx));
#line 2967 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368301_10_non_const_x_to_go <= __cuda_local_var_368302_23_non_const_y_extension)
#line 2968 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2970 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_10_non_const_x_move = __cuda_local_var_368301_10_non_const_x_to_go;
#line 2971 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_22_non_const_y_move = (__cuda_local_var_368295_10_non_const_delta_yx * __cuda_local_var_368301_10_non_const_x_to_go);
#line 2972 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go = (0.080000000000000002);
#line 2974 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go -= (__cuda_local_var_368296_22_non_const_y_move * ((double)(abs(__cuda_local_var_368294_25_non_const_y_move_direction))));
#line 2975 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_7_non_const_voxel_x += __cuda_local_var_368294_7_non_const_x_move_direction;
#line 2976 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368301_19_non_const_y_to_go == (0.0))
#line 2977 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2978 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go = (0.080000000000000002);
#line 2979 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_16_non_const_voxel_y -= __cuda_local_var_368294_25_non_const_y_move_direction;
#line 2980 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2981 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2984 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
else 
#line 2984 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2986 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_10_non_const_x_move = ( fdivide(__cuda_local_var_368301_19_non_const_y_to_go , __cuda_local_var_368295_10_non_const_delta_yx));
#line 2987 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368296_22_non_const_y_move = __cuda_local_var_368301_19_non_const_y_to_go;
#line 2988 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_10_non_const_x_to_go -= (__cuda_local_var_368296_10_non_const_x_move * ((double)(abs(__cuda_local_var_368294_7_non_const_x_move_direction))));
#line 2990 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368301_19_non_const_y_to_go = (0.080000000000000002);
#line 2991 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368303_16_non_const_voxel_y -= __cuda_local_var_368294_25_non_const_y_move_direction;
#line 2992 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 2995 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368305_7_non_const_voxel = ((__cuda_local_var_368303_7_non_const_voxel_x + (__cuda_local_var_368303_16_non_const_voxel_y * 200)) + ((__cuda_local_var_368303_25_non_const_voxel_z * 200) * 200));
#line 2996 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368306_8_non_const_end_walk = ((__nv_bool)((((__cuda_local_var_368305_7_non_const_voxel == __cuda_local_var_368305_14_non_const_voxel_out) || (__cuda_local_var_368303_7_non_const_voxel_x >= 200)) || (__cuda_local_var_368303_16_non_const_voxel_y >= 200)) || (__cuda_local_var_368303_25_non_const_voxel_z >= 24)));
#line 2997 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (!(__cuda_local_var_368306_8_non_const_end_walk)) {
#line 2998 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
_ZN43_INTERNAL_26_pCT_Reconstruction_cpp1_ii_ped9atomicAddEPii((SM_counts + __cuda_local_var_368305_7_non_const_voxel), 1); }
#line 2999 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 3001 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 3002 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 3003 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 3067 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z21SM_edge_detection_GPUPiS_(
#line 3067 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *SM_counts, 
#line 3067 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *SM_threshold){
#line 3068 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3069 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368519_6_non_const_row;
#line 3069 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368519_24_non_const_column;
#line 3069 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368519_45_non_const_slice;
#line 3070 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_368520_8_non_const_x;
#line 3071 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_368521_8_non_const_y;
#line 3072 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368522_6_non_const_voxel;
#line 3069 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368519_6_non_const_row = ((int)(blockIdx.y));
#line 3069 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368519_24_non_const_column = ((int)(blockIdx.x));
#line 3069 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368519_45_non_const_slice = ((int)(threadIdx.x));
#line 3070 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368520_8_non_const_x = ((float)((((double)(__cuda_local_var_368519_24_non_const_column - 100)) + (0.5)) * (0.080000000000000002)));
#line 3071 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368521_8_non_const_y = ((float)((((double)(100 - __cuda_local_var_368519_6_non_const_row)) - (0.5)) * (0.080000000000000002)));
#line 3072 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368522_6_non_const_voxel = ((__cuda_local_var_368519_24_non_const_column + (__cuda_local_var_368519_6_non_const_row * 200)) + ((__cuda_local_var_368519_45_non_const_slice * 200) * 200));
#line 3073 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368522_6_non_const_voxel < 960000)
#line 3074 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2167;
#line 3075 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)(SM_counts[__cuda_local_var_368522_6_non_const_voxel])) > ((1.0) * ((double)(SM_threshold[__cuda_local_var_368519_45_non_const_slice])))) {
#line 3076 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_counts[__cuda_local_var_368522_6_non_const_voxel]) = 1; } else  {
#line 3078 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_counts[__cuda_local_var_368522_6_non_const_voxel]) = 0; }
#line 3079 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)((_Z8_Pow_intIfET_S0_i(__cuda_local_var_368520_8_non_const_x, 2)) + (_Z8_Pow_intIfET_S0_i(__cuda_local_var_368521_8_non_const_y, 2)))) >= ((__T2167 = ((8.0) - ( fdivide((fmax((0.080000000000000002), (0.080000000000000002))) , (2.0))))) , (_Z8_Pow_intIdET_S0_i(__T2167, 2)))) {
#line 3080 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_counts[__cuda_local_var_368522_6_non_const_voxel]) = 0; }
#line 3081 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 3082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 3142 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z23SM_edge_detection_GPU_2PiS_(
#line 3142 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *SM_counts, 
#line 3142 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *SM_differences){
#line 3143 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2168;
#line 3144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368594_6_non_const_row;
#line 3144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368594_24_non_const_column;
#line 3144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368594_45_non_const_slice;
#line 3145 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368595_6_non_const_voxel;
#line 3146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368596_6_non_const_difference;
#line 3146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368596_18_non_const_max_difference;
#line 3161 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368611_6_non_const_slice_threshold;
#line 3173 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_368623_8_non_const_x;
#line 3174 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_368624_8_non_const_y;
#line 3144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368594_6_non_const_row = ((int)(blockIdx.y));
#line 3144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368594_24_non_const_column = ((int)(blockIdx.x));
#line 3144 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368594_45_non_const_slice = ((int)(threadIdx.x));
#line 3145 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368595_6_non_const_voxel = ((__cuda_local_var_368594_24_non_const_column + (__cuda_local_var_368594_6_non_const_row * 200)) + ((__cuda_local_var_368594_45_non_const_slice * 200) * 200));
#line 3146 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368596_18_non_const_max_difference = 0;
#line 3147 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((((__cuda_local_var_368594_6_non_const_row != 0) && (__cuda_local_var_368594_6_non_const_row != 199)) && (__cuda_local_var_368594_24_non_const_column != 0)) && (__cuda_local_var_368594_24_non_const_column != 199))
#line 3148 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{ {
#line 3149 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int current_row;
#line 3149 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
current_row = (__cuda_local_var_368594_6_non_const_row - 1);
#line 3149 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (current_row <= (__cuda_local_var_368594_6_non_const_row + 1)); current_row++)
#line 3150 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{ {
#line 3151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int current_column;
#line 3151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
current_column = (__cuda_local_var_368594_24_non_const_column - 1);
#line 3151 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (current_column <= (__cuda_local_var_368594_24_non_const_column + 1)); current_column++)
#line 3152 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3153 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368596_6_non_const_difference = ((SM_counts[__cuda_local_var_368595_6_non_const_voxel]) - (SM_counts[((current_column + (current_row * 200)) + ((__cuda_local_var_368594_45_non_const_slice * 200) * 200))]));
#line 3154 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368596_6_non_const_difference > __cuda_local_var_368596_18_non_const_max_difference) {
#line 3155 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368596_18_non_const_max_difference = __cuda_local_var_368596_6_non_const_difference; }
#line 3156 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 3157 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 3158 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_differences[__cuda_local_var_368595_6_non_const_voxel]) = __cuda_local_var_368596_18_non_const_max_difference;
#line 3159 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 3160 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"
__syncthreads();
#line 3160 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 3162 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368596_18_non_const_max_difference = 0; {
#line 3163 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int pixel;
#line 3163 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
pixel = 0;
#line 3163 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (pixel < 40000); pixel++)
#line 3164 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3165 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368595_6_non_const_voxel = (pixel + ((__cuda_local_var_368594_45_non_const_slice * 200) * 200));
#line 3166 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if ((SM_differences[__cuda_local_var_368595_6_non_const_voxel]) > __cuda_local_var_368596_18_non_const_max_difference)
#line 3167 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3168 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368596_18_non_const_max_difference = (SM_differences[__cuda_local_var_368595_6_non_const_voxel]);
#line 3169 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368611_6_non_const_slice_threshold = (SM_counts[__cuda_local_var_368595_6_non_const_voxel]);
#line 3170 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 3171 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} }
#line 3172 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v6.0\\include\\device_functions.h"
__syncthreads();
#line 3172 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 3173 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368623_8_non_const_x = ((float)((((double)(__cuda_local_var_368594_24_non_const_column - 100)) + (0.5)) * (0.080000000000000002)));
#line 3174 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368624_8_non_const_y = ((float)((((double)(100 - __cuda_local_var_368594_6_non_const_row)) - (0.5)) * (0.080000000000000002)));
#line 3175 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_368595_6_non_const_voxel < 960000)
#line 3176 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3177 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)(SM_counts[__cuda_local_var_368595_6_non_const_voxel])) > ((1.0) * ((double)__cuda_local_var_368611_6_non_const_slice_threshold))) {
#line 3178 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_counts[__cuda_local_var_368595_6_non_const_voxel]) = 1; } else  {
#line 3180 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_counts[__cuda_local_var_368595_6_non_const_voxel]) = 0; }
#line 3181 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((double)((_Z8_Pow_intIfET_S0_i(__cuda_local_var_368623_8_non_const_x, 2)) + (_Z8_Pow_intIfET_S0_i(__cuda_local_var_368624_8_non_const_y, 2)))) >= ((__T2168 = ((8.0) - ( fdivide((fmax((0.080000000000000002), (0.080000000000000002))) , (2.0))))) , (_Z8_Pow_intIdET_S0_i(__T2168, 2)))) {
#line 3182 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(SM_counts[__cuda_local_var_368595_6_non_const_voxel]) = 0; }
#line 3183 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
} 
#line 3184 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 4223 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z28create_hull_image_hybrid_GPURPbRPf(
#line 4223 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool **x_hull, 
#line 4223 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
float **FBP_image){
#line 4224 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 4225 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_369675_6_non_const_row;
#line 4225 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_369675_24_non_const_column;
#line 4225 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_369675_45_non_const_slice;
#line 4226 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_369676_6_non_const_voxel;
#line 4225 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_369675_6_non_const_row = ((int)(blockIdx.y));
#line 4225 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_369675_24_non_const_column = ((int)(blockIdx.x));
#line 4225 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_369675_45_non_const_slice = ((int)(threadIdx.x));
#line 4226 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_369676_6_non_const_voxel = ((__cuda_local_var_369675_24_non_const_column + (__cuda_local_var_369675_6_non_const_row * 200)) + ((__cuda_local_var_369675_45_non_const_slice * 200) * 200));
#line 4227 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
((*FBP_image)[__cuda_local_var_369676_6_non_const_voxel]) *= ((float)((*x_hull)[__cuda_local_var_369676_6_non_const_voxel])); 
#line 4228 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 4899 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z13test_func_GPUPi(
#line 4899 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *a){
#line 4900 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{  double __T2169;
 double __T2170;
#line 4902 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_370352_9_non_const_delta_yx;
#line 4903 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_370353_9_non_const_x_to_go;
#line 4904 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_370354_9_non_const_y_to_go;
#line 4905 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_370355_9_non_const_y_to_go2;
#line 4906 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_370356_9_non_const_y_move;
#line 4918 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_370368_9_non_const_y;
#line 4928 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_370378_8_non_const_x;
#line 4930 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_370380_8_non_const_z;
#line 4931 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_370381_8_non_const_z2;
#line 4932 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 float __cuda_local_var_370382_8_non_const_z3;
#line 4933 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_370383_7_non_const_less;
#line 4934 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_370384_7_non_const_less2;
#line 4935 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 __nv_bool __cuda_local_var_370385_7_non_const_less3;
#line 4902 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370352_9_non_const_delta_yx = (1.0);
#line 4903 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370353_9_non_const_x_to_go = (0.024);
#line 4904 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370354_9_non_const_y_to_go = (0.014999999999999999);
#line 4905 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370355_9_non_const_y_to_go2 = __cuda_local_var_370354_9_non_const_y_to_go;
#line 4906 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370356_9_non_const_y_move = (__cuda_local_var_370352_9_non_const_delta_yx * __cuda_local_var_370353_9_non_const_x_to_go);
#line 4907 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (1) {
#line 4908 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"-1")); }
#line 4909 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (1) {
#line 4910 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"1")); }
#line 4911 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (0) {
#line 4912 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"0")); }
#line 4913 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370354_9_non_const_y_to_go -= (((double)(!(sin(__cuda_local_var_370352_9_non_const_delta_yx)))) * __cuda_local_var_370356_9_non_const_y_move);
#line 4915 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370355_9_non_const_y_to_go2 -= ((((double)(!(sin(__cuda_local_var_370352_9_non_const_delta_yx)))) * __cuda_local_var_370352_9_non_const_delta_yx) * __cuda_local_var_370353_9_non_const_x_to_go);
#line 4917 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)" delta_yx = %8f y_move = %8f y_to_go = %8f y_to_go2 = %8f\n"), __cuda_local_var_370352_9_non_const_delta_yx, __cuda_local_var_370356_9_non_const_y_move, __cuda_local_var_370354_9_non_const_y_to_go, __cuda_local_var_370355_9_non_const_y_to_go2);
#line 4918 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370368_9_non_const_y = (1.3600000000000001);
#line 4928 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370378_8_non_const_x = (1.0F);
#line 4929 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370368_9_non_const_y = (1.0);
#line 4930 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370380_8_non_const_z = ((float)( fdivide((fabs((2.0))) , ((__T2169 = (((double)__cuda_local_var_370378_8_non_const_x) - __cuda_local_var_370368_9_non_const_y)) , (fabs(__T2169))))));
#line 4931 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370381_8_non_const_z2 = ((float)( fdivide((fabs((-2.0))) , ((__T2170 = (((double)__cuda_local_var_370378_8_non_const_x) - __cuda_local_var_370368_9_non_const_y)) , (fabs(__T2170))))));
#line 4932 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370382_8_non_const_z3 = (__cuda_local_var_370380_8_non_const_z * __cuda_local_var_370378_8_non_const_x);
#line 4933 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370383_7_non_const_less = ((__nv_bool)(__cuda_local_var_370380_8_non_const_z < __cuda_local_var_370381_8_non_const_z2));
#line 4934 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370384_7_non_const_less2 = ((__nv_bool)(__cuda_local_var_370378_8_non_const_x < __cuda_local_var_370380_8_non_const_z));
#line 4935 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_370385_7_non_const_less3 = ((__nv_bool)(__cuda_local_var_370378_8_non_const_x < __cuda_local_var_370381_8_non_const_z2));
#line 4936 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_370383_7_non_const_less) {
#line 4937 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(a[0]) = 1; }
#line 4938 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_370384_7_non_const_less2) {
#line 4939 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(a[1]) = 1; }
#line 4940 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (__cuda_local_var_370385_7_non_const_less3) {
#line 4941 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(a[2]) = 1; }
#line 4943 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
printf(((const char *)"%3f %3f %3f %d %d %d\n"), ((double)__cuda_local_var_370380_8_non_const_z), ((double)__cuda_local_var_370381_8_non_const_z2), ((double)__cuda_local_var_370382_8_non_const_z3), ((int)__cuda_local_var_370383_7_non_const_less), ((int)__cuda_local_var_370384_7_non_const_less2), ((int)__cuda_local_var_370385_7_non_const_less3)); 
#line 4951 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2108 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z19initialize_hull_GPUIbEvPT_(
#line 2108 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *hull){
#line 2109 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367560_6_non_const_row;
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367560_24_non_const_column;
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367560_45_non_const_slice;
#line 2111 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367561_6_non_const_voxel;
#line 2112 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367562_9_non_const_x;
#line 2113 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367563_9_non_const_y;
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367560_6_non_const_row = ((int)(blockIdx.y));
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367560_24_non_const_column = ((int)(blockIdx.x));
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367560_45_non_const_slice = ((int)(threadIdx.x));
#line 2111 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367561_6_non_const_voxel = ((__cuda_local_var_367560_24_non_const_column + (__cuda_local_var_367560_6_non_const_row * 200)) + ((__cuda_local_var_367560_45_non_const_slice * 200) * 200));
#line 2112 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367562_9_non_const_x = ((((double)(__cuda_local_var_367560_24_non_const_column - 100)) + (0.5)) * (0.080000000000000002));
#line 2113 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367563_9_non_const_y = ((((double)(100 - __cuda_local_var_367560_6_non_const_row)) - (0.5)) * (0.080000000000000002));
#line 2114 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((_Z8_Pow_intIdET_S0_i(__cuda_local_var_367562_9_non_const_x, 2)) + (_Z8_Pow_intIdET_S0_i(__cuda_local_var_367563_9_non_const_y, 2))) < (_Z8_Pow_intIdET_S0_i((8.0), 2))) {
#line 2115 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(hull[__cuda_local_var_367561_6_non_const_voxel]) = ((__nv_bool)1); } else  {
#line 2117 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(hull[__cuda_local_var_367561_6_non_const_voxel]) = ((__nv_bool)0); } 
#line 2118 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 2108 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z19initialize_hull_GPUIiEvPT_(
#line 2108 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
int *hull){
#line 2109 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367560_6_non_const_row;
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367560_24_non_const_column;
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367560_45_non_const_slice;
#line 2111 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_367561_6_non_const_voxel;
#line 2112 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367562_9_non_const_x;
#line 2113 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_367563_9_non_const_y;
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367560_6_non_const_row = ((int)(blockIdx.y));
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367560_24_non_const_column = ((int)(blockIdx.x));
#line 2110 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367560_45_non_const_slice = ((int)(threadIdx.x));
#line 2111 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367561_6_non_const_voxel = ((__cuda_local_var_367560_24_non_const_column + (__cuda_local_var_367560_6_non_const_row * 200)) + ((__cuda_local_var_367560_45_non_const_slice * 200) * 200));
#line 2112 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367562_9_non_const_x = ((((double)(__cuda_local_var_367560_24_non_const_column - 100)) + (0.5)) * (0.080000000000000002));
#line 2113 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_367563_9_non_const_y = ((((double)(100 - __cuda_local_var_367560_6_non_const_row)) - (0.5)) * (0.080000000000000002));
#line 2114 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (((_Z8_Pow_intIdET_S0_i(__cuda_local_var_367562_9_non_const_x, 2)) + (_Z8_Pow_intIdET_S0_i(__cuda_local_var_367563_9_non_const_y, 2))) < (_Z8_Pow_intIdET_S0_i((8.0), 2))) {
#line 2115 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(hull[__cuda_local_var_367561_6_non_const_voxel]) = 1; } else  {
#line 2117 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(hull[__cuda_local_var_367561_6_non_const_voxel]) = 0; } 
#line 2118 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}
#line 3265 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__declspec(__global__)  void _Z20averaging_filter_GPUIbEvPT_S1_b(
#line 3265 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *image, 
#line 3265 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool *new_value, 
#line 3265 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__nv_bool is_hull){
#line 3266 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{
#line 3267 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368717_6_non_const_voxel_x;
#line 3268 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368718_6_non_const_voxel_y;
#line 3269 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368719_6_non_const_voxel_z;
#line 3270 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368720_6_non_const_voxel;
#line 3271 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368721_6_non_const_left_edge;
#line 3272 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368722_6_non_const_right_edge;
#line 3273 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368723_6_non_const_top_edge;
#line 3274 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368724_6_non_const_bottom_edge;
#line 3275 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int __cuda_local_var_368725_6_non_const_neighborhood_voxels;
#line 3276 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368726_9_non_const_sum_threshold;
#line 3277 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 double __cuda_local_var_368727_9_non_const_sum;
#line 3267 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368717_6_non_const_voxel_x = ((int)(blockIdx.x));
#line 3268 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368718_6_non_const_voxel_y = ((int)(blockIdx.y));
#line 3269 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368719_6_non_const_voxel_z = ((int)(threadIdx.x));
#line 3270 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368720_6_non_const_voxel = ((__cuda_local_var_368717_6_non_const_voxel_x + (__cuda_local_var_368718_6_non_const_voxel_y * 200)) + ((__cuda_local_var_368719_6_non_const_voxel_z * 200) * 200));
#line 3271 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368721_6_non_const_left_edge = (max((__cuda_local_var_368717_6_non_const_voxel_x - 2), 0));
#line 3272 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368722_6_non_const_right_edge = (min((__cuda_local_var_368717_6_non_const_voxel_x + 2), 199));
#line 3273 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368723_6_non_const_top_edge = (max((__cuda_local_var_368718_6_non_const_voxel_y - 2), 0));
#line 3274 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368724_6_non_const_bottom_edge = (min((__cuda_local_var_368718_6_non_const_voxel_y + 2), 199));
#line 3275 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368725_6_non_const_neighborhood_voxels = (((__cuda_local_var_368722_6_non_const_right_edge - __cuda_local_var_368721_6_non_const_left_edge) + 1) * ((__cuda_local_var_368724_6_non_const_bottom_edge - __cuda_local_var_368723_6_non_const_top_edge) + 1));
#line 3276 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368726_9_non_const_sum_threshold = (((double)__cuda_local_var_368725_6_non_const_neighborhood_voxels) * (0.10000000000000001));
#line 3277 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368727_9_non_const_sum = (0.0); {
#line 3280 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int column;
#line 3280 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
column = __cuda_local_var_368721_6_non_const_left_edge;
#line 3280 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (column <= __cuda_local_var_368722_6_non_const_right_edge); column++) { {
#line 3281 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
 int row;
#line 3281 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
row = __cuda_local_var_368723_6_non_const_top_edge;
#line 3281 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
for (; (row <= __cuda_local_var_368724_6_non_const_bottom_edge); row++) {
#line 3282 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
__cuda_local_var_368727_9_non_const_sum += ((double)(image[((column + (row * 200)) + ((__cuda_local_var_368719_6_non_const_voxel_z * 200) * 200))])); } } } }
#line 3283 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
if (is_hull) {
#line 3284 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(new_value[__cuda_local_var_368720_6_non_const_voxel]) = ((__nv_bool)(__cuda_local_var_368727_9_non_const_sum > __cuda_local_var_368726_9_non_const_sum_threshold)); } else  {
#line 3286 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
(new_value[__cuda_local_var_368720_6_non_const_voxel]) = ((__nv_bool)(( fdivide(__cuda_local_var_368727_9_non_const_sum , ((double)__cuda_local_var_368725_6_non_const_neighborhood_voxels))) != (0.0))); } 
#line 3287 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}}

