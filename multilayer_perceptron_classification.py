#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# $example on$
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession
import mlflow
import hyperopt
from hyperopt import pyll

if __name__ == "__main__":
    spark = SparkSession \
        .builder.appName("multilayer_perceptron_classification_example").getOrCreate()

    # $example on$
    # Load training data
    data = spark.read.format("libsvm") \
        .load("data/mllib/sample_multiclass_classification_data.txt")

    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4], 1234)
    train = splits[0]
    test = splits[1]


    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)

    # MLFLOW, HYPEROPTS SETUP ####################################################################
    # define function for model evaluation
    def evaluate_model(params):
        # train_values = train_inputs.value
        # test_values = test_inputs.value
        print('evaluate_model: get data')
        train_values = train
        test_values = test
        print('evaluate_model: get params')
        layer_left = int(params['layer_left'])
        layer_right = int(params['layer_right'])
        print('evaluate_model: set layers')
        layers = [4, layer_left, layer_right, 3]

        print('create the trainer and set its parameters')
        trainer = MultilayerPerceptronClassifier(maxIter=2, layers=layers, blockSize=128, seed=1234)
        print('train the model')
        model = trainer.fit(train_values)

        print('compute accuracy on the test set')
        result = model.transform(test_values)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        loss = evaluator.evaluate(predictionAndLabels)
        print("Test set accuracy = " + str(loss))
        return {'loss': -loss, 'status': hyperopt.STATUS_OK}


    # print('replicate input_pd dataframe to workers in Spark cluster')
    # train_inputs = spark.sparkContext.broadcast(train)
    # test_inputs = spark.sparkContext.broadcast(test)

    print('define search space')
    search_space = {
        'layer_left': pyll.scope.int(hyperopt.hp.quniform('layer_left', 1, 10, q=1)),
        'layer_right': pyll.scope.int(hyperopt.hp.quniform('layer_right', 1, 10, q=1)),
    }

    print('select optimization algorithm')
    algo = hyperopt.tpe.suggest

    print('configure hyperopt settings to distribute to all executors on workers')
    # spark_trials = hyperopt.SparkTrials(parallelism=2)
    # spark_trials = hyperopt.SparkTrials()
    spark_trials = hyperopt.Trials()

    with mlflow.start_run(run_name='model_run'):
        print('mlflow start')
        best_params = hyperopt.fmin(
            fn=evaluate_model,
            space=search_space,
            algo=algo,
            max_evals=10,
            trials=spark_trials,
        )
    best_model_params = hyperopt.space_eval(search_space, best_params)
    print(best_model_params)
    ######################################################################################

    # $example off$

    spark.stop()
