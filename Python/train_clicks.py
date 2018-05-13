import os
import numpy as np
import joblib
import FF_Demo
from trials import fullforce_poisson_clicks

p = FF_Demo.create_parameters(dt=0.001)
p['g'] = 1.5  # From paper
p['network_size'] = 1000
p['ff_num_batches'] = 50
p['ff_trials_per_batch'] = 20
p['ff_init_trials'] = 10
p['test_init_trials'] = 10
p['ff_steps_per_update'] = 1

rnn = FF_Demo.RNN(p, 2, 1)

rnn.train(fullforce_poisson_clicks, monitor_training=True)

if not os.path.isdir('data'):
    os.makedirs('data')

fn=os.path.join('data', str(int(time.time())) +
                '_fullforce_poisson_rnn.p.z')
joblib.dump(rnn, fn, compress=3)
print("Saved")