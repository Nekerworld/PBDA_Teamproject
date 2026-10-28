import os, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
rng = np.random.default_rng(0)
n = 2_000_000
X = rng.normal(size=(n, 23)); y = (X[:, 0] + rng.normal(size=n) > 2.5).astype(int)
for threads in (1, 8):
    with threadpool_limits(limits=threads, user_api='openmp'):
        m = HistGradientBoostingClassifier(learning_rate=.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=50,
                                           l2_regularization=1., early_stopping=False, random_state=42)
        t = time.perf_counter(); m.fit(X, y); print(threads, 'threads fit', round(time.perf_counter() - t, 1), 's', flush=True)
        p8 = m.predict_proba(X[:100000])[:, 1]
    np.save(f'/tmp/p{threads}.npy', p8)
print('max diff 1 vs 8 threads', np.abs(np.load('/tmp/p1.npy') - np.load('/tmp/p8.npy')).max())
