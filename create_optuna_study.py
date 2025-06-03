import os
import optuna
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', help='optuna storage network host', default='localhost')
parser.add_argument('--port', help='optuna storage network port', default=0)

args = parser.parse_args()
postgres_host = args.host
postgres_port = args.port
pg_pass_file = '/home/'+os.environ["USER"]+'/postgres/config/postgres-password'


# Create an Optuna study
study_name = "pairdiff0"
pruner = None
n_train_iter = 100
with open(pg_pass_file, 'r') as f:
    db_password = f.read().strip()
#pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_train_iter, reduction_factor=3)
#storage = optuna.storages.RDBStorage(url="postgresql://mhan:"+db_password+"@val:"+str(postgres_port)+"/example")
#storage = optuna.storages.RDBStorage(url="postgresql://mhan:"+db_password+"@localhost"+"/example")
storage = optuna.storages.RDBStorage(url="postgresql://mhan:"+db_password+"@"+postgres_host+":"+str(postgres_port)+"/example")
#optuna.delete_study(study_name=study_name, storage=storage)
study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, pruner=pruner, load_if_exists=True)



