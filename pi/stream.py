import numpy as np, pandas as pd, time, os, sys
sys.path.insert(0,os.path.expanduser('~/inference_engine/src'))
from infer import infer

df    =pd.read_csv(os.path.expanduser('~/inference_engine/data/ETTh1_anomaly.csv'))
data  =df.drop(columns=['date']).values.astype(np.float32)
labels=np.load(os.path.expanduser('~/inference_engine/data/labels.npy'))

SCALER_MEAN =np.load(os.path.expanduser('~/inference_engine/weights/scaler_mean.npy'))
SCALER_SCALE=np.load(os.path.expanduser('~/inference_engine/weights/scaler_scale.npy'))

SEQ_LEN=336; PRED_LEN=96; N=200
THRESHOLD=0.5319
import numpy as np, pandas as pd, time, os, sys
sys.path.insert(0,os.path.expanduser('~/inference_engine/src'))
from infer import infer

df    =pd.read_csv(os.path.expanduser('~/inference_engine/data/ETTh1_anomaly.csv'))
data  =df.drop(columns=['date']).values.astype(np.float32)
labels=np.load(os.path.expanduser('~/inference_engine/data/labels.npy'))

SCALER_MEAN =np.load(os.path.expanduser('~/inference_engine/weights/scaler_mean.npy'))
SCALER_SCALE=np.load(os.path.expanduser('~/inference_engine/weights/scaler_scale.npy'))

SEQ_LEN=336; PRED_LEN=96; N=200
THRESHOLD=0.5319
import numpy as np, pandas as pd, time, os, sys
sys.path.insert(0,os.path.expanduser('~/inference_engine/src'))
from infer import infer

df    =pd.read_csv(os.path.expanduser('~/inference_engine/data/ETTh1_anomaly.csv'))
data  =df.drop(columns=['date']).values.astype(np.float32)
labels=np.load(os.path.expanduser('~/inference_engine/data/labels.npy'))

SCALER_MEAN =np.load(os.path.expanduser('~/inference_engine/weights/scaler_mean.npy'))
SCALER_SCALE=np.load(os.path.expanduser('~/inference_engine/weights/scaler_scale.npy'))

SEQ_LEN=336; PRED_LEN=96; N=200
THRESHOLD=0.5319

test_start=int(0.8*len(data))
latencies=[]
for i in range(N):
    idx=test_start+i
    t0=time.perf_counter()
    pred=infer(data[idx:idx+SEQ_LEN])
    lat=(time.perf_counter()-t0)*1000
    latencies.append(lat)
    target_norm=(data[idx+SEQ_LEN:idx+SEQ_LEN+PRED_LEN]-SCALER_MEAN)/SCALER_SCALE
    mse=float(np.mean((pred-target_norm)**2))
    tl=int(labels[idx+SEQ_LEN-1]); pl=int(mse>THRESHOLD)
    tag='' if tl==pl else ('[FN]' if tl else '[FP]')
    print(f'W{i:03d}: {lat:6.1f}ms  MSE={mse:.4f}  {tag}')
print(f'Mean: {np.mean(latencies):.0f}ms  P95: {np.percentile(latencies,95):.0f}ms')
