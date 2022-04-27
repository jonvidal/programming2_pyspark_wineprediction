import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

conf = pyspark.SparkConf().setAppName('Pyspark_Prog2').setMaster('local')
conf.set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.2')
sc = pyspark.SparkContext.getOrCreate(conf)
spark = SparkSession(sc)

path = sys.argv[1]
df = spark.read.format("csv").load(path, delimiter=';', header='True', inferSchema='True')
qty = '""""quality"""""'
features = np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select(qty).collect())

labeled_points = []
for x, y in zip(features, label):
    lp = LabeledPoint(y, x)
    labeled_points.append(lp)
dataset = sc.parallelize(labeled_points)

model = RandomForestModel.load(sc, "Trained_RFModel/")

dataset_map=dataset.map(lambda x: x.features)
pred = model.predict(dataset_map)
lp = dataset.map(lambda x: x.label).zip(pred)
lp_res = lp.toDF([qty, "Prediction"])
lp_res_df = lp_res.toPandas()

f1score = f1_score(lp_res_df[qty], lp_res_df['Prediction'], average='micro')
accuracy = accuracy_score(lp_res_df[qty], lp_res_df['Prediction'])
p_score = precision_score(lp_res_df[qty], lp_res_df['Prediction'], average='micro')
r_score = recall_score(lp_res_df[qty], lp_res_df['Prediction'], average='micro')

print("F1 score: %.3f" % f1score)
print('Precision: %.3f' % p_score)
print('Recall: %.3f' % r_score)
print("Accuracy: %.3f" % accuracy)