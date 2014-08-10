#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "pCT_Reconstruction.fatbin.c"
extern void __device_stub__Z30recon_volume_intersections_GPUiPiPbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(int, int *, bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z11binning_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z19calculate_means_GPUPiPfS0_S0_(int *, float *, float *, float *);
extern void __device_stub__Z26sum_squared_deviations_GPUiPiPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_(int, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z33calculate_standard_deviations_GPUPiPfS0_S0_(int *, float *, float *, float *);
extern void __device_stub__Z20statistical_cuts_GPUiPiS_PfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_Pb(int, int *, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, bool *);
extern void __device_stub__Z22construct_sinogram_GPUPiPf(int *, float *);
extern void __device_stub__Z10filter_GPUPfS_(float *, float *);
extern void __device_stub__Z18backprojection_GPUPfS_(float *, float *);
extern void __device_stub__Z20FBP_image_2_hull_GPUPfPb(float *, bool *);
extern void __device_stub__Z17carve_differencesPiS_(int *, int *);
extern void __device_stub__Z6SC_GPUiPbPiS_PfS1_S1_S1_S1_S1_S1_(const int, bool *, int *, bool *, float *, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z7MSC_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_(const int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z22MSC_edge_detection_GPUPi(int *);
extern void __device_stub__Z6SM_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_(const int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z21SM_edge_detection_GPUPiS_(int *, int *);
extern void __device_stub__Z23SM_edge_detection_GPU_2PiS_(int *, int *);
extern void __device_stub__Z28create_hull_image_hybrid_GPURPbRPf(bool **, float **);
extern void __device_stub__Z13test_func_GPUPi(int *);
static void __device_stub__Z19initialize_hull_GPUIbEvPT_(bool *);
static void __device_stub__Z19initialize_hull_GPUIiEvPT_(int *);
static void __device_stub__Z20averaging_filter_GPUIbEvPT_S1_b(bool *, bool *, bool);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_26_pCT_Reconstruction_cpp1_ii_ped(void);
#pragma section(".CRT$XCU",read)
__declspec(allocate(".CRT$XCU"))static void (*__dummy_static_init__sti____cudaRegisterAll_26_pCT_Reconstruction_cpp1_ii_ped[])(void) = {__sti____cudaRegisterAll_26_pCT_Reconstruction_cpp1_ii_ped};
void __device_stub__Z30recon_volume_intersections_GPUiPiPbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_(int __par0, int *__par1, bool *__par2, float *__par3, float *__par4, float *__par5, float *__par6, float *__par7, float *__par8, float *__par9, float *__par10, float *__par11, float *__par12, float *__par13, float *__par14, float *__par15, float *__par16, float *__par17, float *__par18, float *__par19, float *__par20, float *__par21, float *__par22, float *__par23, float *__par24){__cudaSetupArgSimple(__par0, 0Ui64);__cudaSetupArgSimple(__par1, 8Ui64);__cudaSetupArgSimple(__par2, 16Ui64);__cudaSetupArgSimple(__par3, 24Ui64);__cudaSetupArgSimple(__par4, 32Ui64);__cudaSetupArgSimple(__par5, 40Ui64);__cudaSetupArgSimple(__par6, 48Ui64);__cudaSetupArgSimple(__par7, 56Ui64);__cudaSetupArgSimple(__par8, 64Ui64);__cudaSetupArgSimple(__par9, 72Ui64);__cudaSetupArgSimple(__par10, 80Ui64);__cudaSetupArgSimple(__par11, 88Ui64);__cudaSetupArgSimple(__par12, 96Ui64);__cudaSetupArgSimple(__par13, 104Ui64);__cudaSetupArgSimple(__par14, 112Ui64);__cudaSetupArgSimple(__par15, 120Ui64);__cudaSetupArgSimple(__par16, 128Ui64);__cudaSetupArgSimple(__par17, 136Ui64);__cudaSetupArgSimple(__par18, 144Ui64);__cudaSetupArgSimple(__par19, 152Ui64);__cudaSetupArgSimple(__par20, 160Ui64);__cudaSetupArgSimple(__par21, 168Ui64);__cudaSetupArgSimple(__par22, 176Ui64);__cudaSetupArgSimple(__par23, 184Ui64);__cudaSetupArgSimple(__par24, 192Ui64);__cudaLaunch(((char *)((void ( *)(int, int *, bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *))recon_volume_intersections_GPU)));}
#line 1085 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void recon_volume_intersections_GPU( int __cuda_0,int *__cuda_1,bool *__cuda_2,float *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10,float *__cuda_11,float *__cuda_12,float *__cuda_13,float *__cuda_14,float *__cuda_15,float *__cuda_16,float *__cuda_17,float *__cuda_18,float *__cuda_19,float *__cuda_20,float *__cuda_21,float *__cuda_22,float *__cuda_23,float *__cuda_24)
#line 1091 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z30recon_volume_intersections_GPUiPiPbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12,__cuda_13,__cuda_14,__cuda_15,__cuda_16,__cuda_17,__cuda_18,__cuda_19,__cuda_20,__cuda_21,__cuda_22,__cuda_23,__cuda_24);
#line 1208 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z11binning_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_( int __par0,  int *__par1,  int *__par2,  bool *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  float *__par8,  float *__par9,  float *__par10,  float *__par11,  float *__par12,  float *__par13,  float *__par14,  float *__par15,  float *__par16,  float *__par17) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 32Ui64); __cudaSetupArgSimple(__par5, 40Ui64); __cudaSetupArgSimple(__par6, 48Ui64); __cudaSetupArgSimple(__par7, 56Ui64); __cudaSetupArgSimple(__par8, 64Ui64); __cudaSetupArgSimple(__par9, 72Ui64); __cudaSetupArgSimple(__par10, 80Ui64); __cudaSetupArgSimple(__par11, 88Ui64); __cudaSetupArgSimple(__par12, 96Ui64); __cudaSetupArgSimple(__par13, 104Ui64); __cudaSetupArgSimple(__par14, 112Ui64); __cudaSetupArgSimple(__par15, 120Ui64); __cudaSetupArgSimple(__par16, 128Ui64); __cudaSetupArgSimple(__par17, 136Ui64); __cudaLaunch(((char *)((void ( *)(int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *))binning_GPU))); }
#line 1407 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void binning_GPU( int __cuda_0,int *__cuda_1,int *__cuda_2,bool *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10,float *__cuda_11,float *__cuda_12,float *__cuda_13,float *__cuda_14,float *__cuda_15,float *__cuda_16,float *__cuda_17)
#line 1414 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z11binning_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12,__cuda_13,__cuda_14,__cuda_15,__cuda_16,__cuda_17);
#line 1478 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z19calculate_means_GPUPiPfS0_S0_( int *__par0,  float *__par1,  float *__par2,  float *__par3) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaLaunch(((char *)((void ( *)(int *, float *, float *, float *))calculate_means_GPU))); }
#line 1514 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void calculate_means_GPU( int *__cuda_0,float *__cuda_1,float *__cuda_2,float *__cuda_3)
#line 1515 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z19calculate_means_GPUPiPfS0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
#line 1524 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z26sum_squared_deviations_GPUiPiPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_( int __par0,  int *__par1,  float *__par2,  float *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  float *__par8,  float *__par9,  float *__par10,  float *__par11,  float *__par12) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 32Ui64); __cudaSetupArgSimple(__par5, 40Ui64); __cudaSetupArgSimple(__par6, 48Ui64); __cudaSetupArgSimple(__par7, 56Ui64); __cudaSetupArgSimple(__par8, 64Ui64); __cudaSetupArgSimple(__par9, 72Ui64); __cudaSetupArgSimple(__par10, 80Ui64); __cudaSetupArgSimple(__par11, 88Ui64); __cudaSetupArgSimple(__par12, 96Ui64); __cudaLaunch(((char *)((void ( *)(int, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *))sum_squared_deviations_GPU))); }
#line 1568 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void sum_squared_deviations_GPU( int __cuda_0,int *__cuda_1,float *__cuda_2,float *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10,float *__cuda_11,float *__cuda_12)
#line 1574 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z26sum_squared_deviations_GPUiPiPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12);
#line 1597 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z33calculate_standard_deviations_GPUPiPfS0_S0_( int *__par0,  float *__par1,  float *__par2,  float *__par3) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaLaunch(((char *)((void ( *)(int *, float *, float *, float *))calculate_standard_deviations_GPU))); }
#line 1609 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void calculate_standard_deviations_GPU( int *__cuda_0,float *__cuda_1,float *__cuda_2,float *__cuda_3)
#line 1610 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z33calculate_standard_deviations_GPUPiPfS0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
#line 1622 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z20statistical_cuts_GPUiPiS_PfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_Pb( int __par0,  int *__par1,  int *__par2,  float *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  float *__par8,  float *__par9,  float *__par10,  float *__par11,  float *__par12,  float *__par13,  float *__par14,  bool *__par15) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 32Ui64); __cudaSetupArgSimple(__par5, 40Ui64); __cudaSetupArgSimple(__par6, 48Ui64); __cudaSetupArgSimple(__par7, 56Ui64); __cudaSetupArgSimple(__par8, 64Ui64); __cudaSetupArgSimple(__par9, 72Ui64); __cudaSetupArgSimple(__par10, 80Ui64); __cudaSetupArgSimple(__par11, 88Ui64); __cudaSetupArgSimple(__par12, 96Ui64); __cudaSetupArgSimple(__par13, 104Ui64); __cudaSetupArgSimple(__par14, 112Ui64); __cudaSetupArgSimple(__par15, 120Ui64); __cudaLaunch(((char *)((void ( *)(int, int *, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, bool *))statistical_cuts_GPU))); }
#line 1702 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void statistical_cuts_GPU( int __cuda_0,int *__cuda_1,int *__cuda_2,float *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10,float *__cuda_11,float *__cuda_12,float *__cuda_13,float *__cuda_14,bool *__cuda_15)
#line 1710 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z20statistical_cuts_GPUiPiS_PfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_Pb( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12,__cuda_13,__cuda_14,__cuda_15);
#line 1735 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z22construct_sinogram_GPUPiPf( int *__par0,  float *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(int *, float *))construct_sinogram_GPU))); }
#line 1766 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void construct_sinogram_GPU( int *__cuda_0,float *__cuda_1)
#line 1767 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z22construct_sinogram_GPUPiPf( __cuda_0,__cuda_1);




}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z10filter_GPUPfS_( float *__par0,  float *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(float *, float *))filter_GPU))); }
#line 1839 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void filter_GPU( float *__cuda_0,float *__cuda_1)
#line 1840 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z10filter_GPUPfS_( __cuda_0,__cuda_1);
#line 1871 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z18backprojection_GPUPfS_( float *__par0,  float *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(float *, float *))backprojection_GPU))); }
#line 1957 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void backprojection_GPU( float *__cuda_0,float *__cuda_1)
#line 1958 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z18backprojection_GPUPfS_( __cuda_0,__cuda_1);
#line 2032 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z20FBP_image_2_hull_GPUPfPb( float *__par0,  bool *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(float *, bool *))FBP_image_2_hull_GPU))); }
#line 2080 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void FBP_image_2_hull_GPU( float *__cuda_0,bool *__cuda_1)
#line 2081 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z20FBP_image_2_hull_GPUPfPb( __cuda_0,__cuda_1);
#line 2091 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z17carve_differencesPiS_( int *__par0,  int *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(int *, int *))carve_differences))); }
#line 2331 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void carve_differences( int *__cuda_0,int *__cuda_1)
#line 2332 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z17carve_differencesPiS_( __cuda_0,__cuda_1);
#line 2349 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z6SC_GPUiPbPiS_PfS1_S1_S1_S1_S1_S1_( const int __par0,  bool *__par1,  int *__par2,  bool *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  float *__par8,  float *__par9,  float *__par10) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 32Ui64); __cudaSetupArgSimple(__par5, 40Ui64); __cudaSetupArgSimple(__par6, 48Ui64); __cudaSetupArgSimple(__par7, 56Ui64); __cudaSetupArgSimple(__par8, 64Ui64); __cudaSetupArgSimple(__par9, 72Ui64); __cudaSetupArgSimple(__par10, 80Ui64); __cudaLaunch(((char *)((void ( *)(const int, bool *, int *, bool *, float *, float *, float *, float *, float *, float *, float *))SC_GPU))); }
#line 2364 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void SC_GPU( const int __cuda_0,bool *__cuda_1,int *__cuda_2,bool *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10)
#line 2369 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z6SC_GPUiPbPiS_PfS1_S1_S1_S1_S1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
#line 2598 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z7MSC_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_( const int __par0,  int *__par1,  int *__par2,  bool *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  float *__par8,  float *__par9,  float *__par10) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 32Ui64); __cudaSetupArgSimple(__par5, 40Ui64); __cudaSetupArgSimple(__par6, 48Ui64); __cudaSetupArgSimple(__par7, 56Ui64); __cudaSetupArgSimple(__par8, 64Ui64); __cudaSetupArgSimple(__par9, 72Ui64); __cudaSetupArgSimple(__par10, 80Ui64); __cudaLaunch(((char *)((void ( *)(const int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *))MSC_GPU))); }
#line 2610 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void MSC_GPU( const int __cuda_0,int *__cuda_1,int *__cuda_2,bool *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10)
#line 2615 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z7MSC_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
#line 2782 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z22MSC_edge_detection_GPUPi( int *__par0) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaLaunch(((char *)((void ( *)(int *))MSC_edge_detection_GPU))); }
#line 2793 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void MSC_edge_detection_GPU( int *__cuda_0)
#line 2794 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z22MSC_edge_detection_GPUPi( __cuda_0);
#line 2820 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z6SM_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_( const int __par0,  int *__par1,  int *__par2,  bool *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  float *__par8,  float *__par9,  float *__par10) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 32Ui64); __cudaSetupArgSimple(__par5, 40Ui64); __cudaSetupArgSimple(__par6, 48Ui64); __cudaSetupArgSimple(__par7, 56Ui64); __cudaSetupArgSimple(__par8, 64Ui64); __cudaSetupArgSimple(__par9, 72Ui64); __cudaSetupArgSimple(__par10, 80Ui64); __cudaLaunch(((char *)((void ( *)(const int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *))SM_GPU))); }
#line 2832 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void SM_GPU( const int __cuda_0,int *__cuda_1,int *__cuda_2,bool *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,float *__cuda_8,float *__cuda_9,float *__cuda_10)
#line 2837 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z6SM_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
#line 3003 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z21SM_edge_detection_GPUPiS_( int *__par0,  int *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(int *, int *))SM_edge_detection_GPU))); }
#line 3067 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void SM_edge_detection_GPU( int *__cuda_0,int *__cuda_1)
#line 3068 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z21SM_edge_detection_GPUPiS_( __cuda_0,__cuda_1);
#line 3082 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z23SM_edge_detection_GPU_2PiS_( int *__par0,  int *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(int *, int *))SM_edge_detection_GPU_2))); }
#line 3142 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void SM_edge_detection_GPU_2( int *__cuda_0,int *__cuda_1)
#line 3143 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z23SM_edge_detection_GPU_2PiS_( __cuda_0,__cuda_1);
#line 3184 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z28create_hull_image_hybrid_GPURPbRPf( bool **__par0,  float **__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(bool *&, float *&))create_hull_image_hybrid_GPU))); }
#line 4223 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void create_hull_image_hybrid_GPU( bool *&__cuda_0,float *&__cuda_1)
#line 4224 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z28create_hull_image_hybrid_GPURPbRPf( __cudaAddressOf(__cuda_0),__cudaAddressOf(__cuda_1));



}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
void __device_stub__Z13test_func_GPUPi( int *__par0) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaLaunch(((char *)((void ( *)(int *))test_func_GPU))); }
#line 4899 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
void test_func_GPU( int *__cuda_0)
#line 4900 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
{__device_stub__Z13test_func_GPUPi( __cuda_0);
#line 4951 "C:/Users/Blake/Documents/GitHub/pct-reconstruction/pCT_Reconstruction.cu"
}
#line 1 "x64/Release/pCT_Reconstruction.cudafe1.stub.c"
static void __device_stub__Z19initialize_hull_GPUIbEvPT_( bool *__par0) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaLaunch(((char *)((void ( *)(bool *))initialize_hull_GPU<bool> ))); }
template<> __specialization_static void __wrapper__device_stub_initialize_hull_GPU<bool>( bool *&__cuda_0){__device_stub__Z19initialize_hull_GPUIbEvPT_( __cuda_0);}
static void __device_stub__Z19initialize_hull_GPUIiEvPT_( int *__par0) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaLaunch(((char *)((void ( *)(int *))initialize_hull_GPU<int> ))); }
template<> __specialization_static void __wrapper__device_stub_initialize_hull_GPU<int>( int *&__cuda_0){__device_stub__Z19initialize_hull_GPUIiEvPT_( __cuda_0);}
static void __device_stub__Z20averaging_filter_GPUIbEvPT_S1_b( bool *__par0,  bool *__par1,  bool __par2) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaLaunch(((char *)((void ( *)(bool *, bool *, bool))averaging_filter_GPU<bool> ))); }
template<> __specialization_static void __wrapper__device_stub_averaging_filter_GPU<bool>( bool *&__cuda_0,bool *&__cuda_1,bool &__cuda_2){__device_stub__Z20averaging_filter_GPUIbEvPT_S1_b( __cuda_0,__cuda_1,__cuda_2);}
static void __nv_cudaEntityRegisterCallback( void **__T2711) {  __nv_dummy_param_ref(__T2711); __nv_save_fatbinhandle_for_managed_rt(__T2711); __cudaRegisterEntry(__T2711, ((void ( *)(bool *, bool *, bool))averaging_filter_GPU<bool> ), _Z20averaging_filter_GPUIbEvPT_S1_b, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *))initialize_hull_GPU<int> ), _Z19initialize_hull_GPUIiEvPT_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(bool *))initialize_hull_GPU<bool> ), _Z19initialize_hull_GPUIbEvPT_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *))test_func_GPU), _Z13test_func_GPUPi, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(bool *&, float *&))create_hull_image_hybrid_GPU), _Z28create_hull_image_hybrid_GPURPbRPf, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *, int *))SM_edge_detection_GPU_2), _Z23SM_edge_detection_GPU_2PiS_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *, int *))SM_edge_detection_GPU), _Z21SM_edge_detection_GPUPiS_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(const int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *))SM_GPU), _Z6SM_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *))MSC_edge_detection_GPU), _Z22MSC_edge_detection_GPUPi, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(const int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *))MSC_GPU), _Z7MSC_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(const int, bool *, int *, bool *, float *, float *, float *, float *, float *, float *, float *))SC_GPU), _Z6SC_GPUiPbPiS_PfS1_S1_S1_S1_S1_S1_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *, int *))carve_differences), _Z17carve_differencesPiS_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(float *, bool *))FBP_image_2_hull_GPU), _Z20FBP_image_2_hull_GPUPfPb, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(float *, float *))backprojection_GPU), _Z18backprojection_GPUPfS_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(float *, float *))filter_GPU), _Z10filter_GPUPfS_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *, float *))construct_sinogram_GPU), _Z22construct_sinogram_GPUPiPf, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int, int *, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, bool *))statistical_cuts_GPU), _Z20statistical_cuts_GPUiPiS_PfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_Pb, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *, float *, float *, float *))calculate_standard_deviations_GPU), _Z33calculate_standard_deviations_GPUPiPfS0_S0_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int, int *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *))sum_squared_deviations_GPU), _Z26sum_squared_deviations_GPUiPiPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int *, float *, float *, float *))calculate_means_GPU), _Z19calculate_means_GPUPiPfS0_S0_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int, int *, int *, bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *))binning_GPU), _Z11binning_GPUiPiS_PbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_, (-1)); __cudaRegisterEntry(__T2711, ((void ( *)(int, int *, bool *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *))recon_volume_intersections_GPU), _Z30recon_volume_intersections_GPUiPiPbPfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_, (-1)); }
static void __sti____cudaRegisterAll_26_pCT_Reconstruction_cpp1_ii_ped(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }
