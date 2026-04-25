#include <torch/extension.h>
#include <cuda_runtime.h>
#define TL 16
__global__ void fqkv_k(const float* X,const float* WQ,const float* WK,const float* WV,
    float* Q,float* K,float* V,int N,int dm,int dk){
    int b=blockIdx.z,row=blockIdx.y*TL+threadIdx.y,col=blockIdx.x*TL+threadIdx.x;
    if(row>=N||col>=dk) return;
    __shared__ float Xs[TL][TL];
    float aq=0,ak=0,av=0;
    for(int t=0;t<(dm+TL-1)/TL;t++){
        int k=t*TL+threadIdx.x;
        Xs[threadIdx.y][threadIdx.x]=(row<N&&k<dm)?X[b*N*dm+row*dm+k]:0;
        __syncthreads();
        for(int i=0;i<TL;i++){
            int gk=t*TL+i;
            if(gk<dm){
                float xv=Xs[threadIdx.y][i];
                aq+=xv*WQ[gk*dk+col];
                ak+=xv*WK[gk*dk+col];
                av+=xv*WV[gk*dk+col];
            }
        }
        __syncthreads();
    }
    Q[b*N*dk+row*dk+col]=aq; K[b*N*dk+row*dk+col]=ak; V[b*N*dk+row*dk+col]=av;
}
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> fused_qkv(torch::Tensor X,torch::Tensor WQ,torch::Tensor WK,torch::Tensor WV){
    int B=X.size(0),N=X.size(1),dm=X.size(2),dk=WQ.size(1);
    auto Q=torch::empty({B,N,dk},X.options());
    auto K=torch::empty({B,N,dk},X.options());
    auto V=torch::empty({B,N,dk},X.options());
    dim3 bl((dk+TL-1)/TL,(N+TL-1)/TL,B);
    fqkv_k<<<bl,dim3(TL,TL)>>>(X.data_ptr<float>(),WQ.data_ptr<float>(),
        WK.data_ptr<float>(),WV.data_ptr<float>(),Q.data_ptr<float>(),
        K.data_ptr<float>(),V.data_ptr<float>(),N,dm,dk);
    return {Q,K,V};
}