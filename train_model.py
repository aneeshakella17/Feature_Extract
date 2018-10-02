from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

db = "/storage/features.hdf5";
model = "imagenet.h5";
jobs = -1;

db = h5py.File(db, "r");
i = int(db["labels"].shape[0] * 75);

print("[INFO] tuning hyperparameters")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv = 3, n_jobs = jobs);
model.fit(db["features"][:i], db["labels"][:i]);
print("[INFO] best hyperparameters : {}".format(model.best_params_));

print("[INFO] evaluating ...")
preds = model.predict(db["features"][i:]);
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]));

print("[INFO] saving model ...");

f = open(model, "wb");
f.write(pickle.dumps(model.best_estimator_));
f.close();
db.close();

