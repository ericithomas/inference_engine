import numpy as np, os
W=os.path.expanduser('~/inference_engine/weights')
ld=lambda n: np.load(f'{W}/{n}.npy')

def layer_norm(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); s=x.std(-1,keepdims=True)
    return w*(x-m)/(s+eps)+b

def gelu(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def softmax(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

def self_attn(x,i,nh=4):
    N,d=x.shape; dk=d//nh
    qkv=x@ld(f'l{i}_attn_inw').T+ld(f'l{i}_attn_inb')
    Q,K,V=np.split(qkv,3,axis=-1)
    Q=Q.reshape(N,nh,dk).transpose(1,0,2)
    K=K.reshape(N,nh,dk).transpose(1,0,2)
    V=V.reshape(N,nh,dk).transpose(1,0,2)
    S=np.matmul(Q,K.transpose(0,2,1))*dk**-0.5
    O=np.matmul(softmax(S),V).transpose(1,0,2).reshape(N,d)
    return O@ld(f'l{i}_attn_outw').T+ld(f'l{i}_attn_outb')

def tl(x,i):
    x=layer_norm(x+self_attn(x,i),ld(f'l{i}_n1w'),ld(f'l{i}_n1b'))
    h=gelu(x@ld(f'l{i}_ff1w').T+ld(f'l{i}_ff1b'))
    return layer_norm(x+h@ld(f'l{i}_ff2w').T+ld(f'l{i}_ff2b'),ld(f'l{i}_n2w'),ld(f'l{i}_n2b'))

def infer(raw,pl=16,st=8,nl=2,nc=7):
    x=(raw-ld('scaler_mean'))/ld('scaler_scale')
    pw=ld('patch_pw'); pb=ld('patch_pb'); pos=ld('pos')
    preds=[]
    for c in range(nc):
        patches=np.stack([x[i:i+pl,c] for i in range(0,len(x)-pl+1,st)])
        z=patches@pw.T+pb+pos
        for i in range(nl): z=tl(z,i)
        preds.append(z.flatten()@ld('head_w').T+ld('head_b'))
    return np.stack(preds,axis=-1)
