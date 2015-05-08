#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Downloads the spamassassin dataset, trains and tests a classifier.
#
# To run:  change into the mahout directory and type:
# examples/bin/classify-spamassassin.sh

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script runs SGD and Bayes classifiers over the classic 20 News Groups."
  exita
fi

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`

# Set commands for dfs
source ${START_PATH}/set-dfs-commands.sh

WORK_DIR=/tmp/mahout-work-${USER}
algorithm=( cnaivebayes-MapReduce naivebayes-MapReduce cnaivebayes-Spark naivebayes-Spark sgd clean)
if [ -n "$1" ]; then
  choice=$1
else
  echo "Please select a number to choose the corresponding task to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  echo "4. ${algorithm[3]}"
  echo "5. ${algorithm[4]}"
  echo "6. ${algorithm[5]}-- cleans up the work area in $WORK_DIR"
  read -p "Enter your choice : " choice
fi

echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
alg=${algorithm[$choice-1]}

# Spark specific check and work
if [ "x$alg" == "xnaivebayes-Spark" -o "x$alg" == "xcnaivebayes-Spark" ]; then
  if [ "$MASTER" == "" ] ; then
    echo "Please set your MASTER env variable to point to your Spark Master URL. exiting..."
    exit 1
  fi
  if [ "$MAHOUT_LOCAL" != "" ] ; then
    echo "Options 3 and 4 can not run in MAHOUT_LOCAL mode. exiting..."
    exit 1
  fi
fi

cd $START_PATH
cd ../..

set -e

if  ( [ "x$alg" == "xnaivebayes-MapReduce" ] ||  [ "x$alg" == "xcnaivebayes-MapReduce" ] || [ "x$alg" == "xnaivebayes-Spark"  ] || [ "x$alg" == "xcnaivebayes-Spark" ] ); then
  c=""

  if [ "x$alg" == "xcnaivebayes-MapReduce" -o "x$alg" == "xnaivebayes-Spark" ]; then
    c=" -c"
  fi

  set -x
  echo "Preparing spamassassin data"
  rm -rf ${WORK_DIR}/spamassassin-all
  mkdir ${WORK_DIR}/spamassassin-all
  cp -R ${WORK_DIR}/spamassassin-bayesinput/* ${WORK_DIR}/spamassassin-all

  if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
    echo "Copying spamassassin data to HDFS"
    set +e
    $DFSRM ${WORK_DIR}/spamassassin-all
    $DFS -mkdir ${WORK_DIR}
    $DFS -mkdir ${WORK_DIR}/spamassassin-all
    set -e
    if [ $HVERSION -eq "1" ] ; then
      echo "Copying spamassassin data to Hadoop 1 HDFS"
      $DFS -put ${WORK_DIR}/spamassassin-all ${WORK_DIR}/spamassassin-all
    elif [ $HVERSION -eq "2" ] ; then
      echo "Copying spamassassin data to Hadoop 2 HDFS"
      $DFS -put ${WORK_DIR}/spamassassin-all ${WORK_DIR}/
    fi
  fi

  echo "Creating sequence files from spamassassin data"
  ./bin/mahout seqdirectory \
    -i ${WORK_DIR}/spamassassin-all \
    -o ${WORK_DIR}/spamassassin-seq -ow

  echo "Converting sequence files to vectors"
  ./bin/mahout seq2sparse \
    -i ${WORK_DIR}/spamassassin-seq \
    -o ${WORK_DIR}/spamassassin-vectors  -lnorm -nv  -wt tfidf

  echo "Creating training and holdout set with a random 80-20 split of the generated vector dataset"
  ./bin/mahout split \
    -i ${WORK_DIR}/spamassassin-vectors/tfidf-vectors \
    --trainingOutput ${WORK_DIR}/spamassassin-train-vectors \
    --testOutput ${WORK_DIR}/spamassassin-test-vectors  \
    --randomSelectionPct 40 --overwrite --sequenceFiles -xm sequential

    if [ "x$alg" == "xnaivebayes-MapReduce"  -o  "x$alg" == "xcnaivebayes-MapReduce" ]; then

      echo "Training Naive Bayes model"
      ./bin/mahout trainnb \
        -i ${WORK_DIR}/spamassassin-train-vectors \
        -o ${WORK_DIR}/model \
        -li ${WORK_DIR}/labelindex \
        -ow $c

      echo "Self testing on training set"

      ./bin/mahout testnb \
        -i ${WORK_DIR}/spamassassin-train-vectors\
        -m ${WORK_DIR}/model \
        -l ${WORK_DIR}/labelindex \
        -ow -o ${WORK_DIR}/spamassassin-testing $c

      echo "Testing on holdout set"

      ./bin/mahout testnb \
        -i ${WORK_DIR}/spamassassin-test-vectors\
        -m ${WORK_DIR}/model \
        -l ${WORK_DIR}/labelindex \
        -ow -o ${WORK_DIR}/spamassassin-testing $c

    elif [ "x$alg" == "xnaivebayes-Spark" -o "x$alg" == "xcnaivebayes-Spark" ]; then

      echo "Training Naive Bayes model"
      ./bin/mahout spark-trainnb \
        -i ${WORK_DIR}/spamassassin-train-vectors \
        -o ${WORK_DIR}/spark-model $c -ow -ma $MASTER

      echo "Self testing on training set"
      ./bin/mahout spark-testnb \
        -i ${WORK_DIR}/spamassassin-train-vectors\
        -m ${WORK_DIR}/spark-model $c -ma $MASTER

      echo "Testing on holdout set"
      ./bin/mahout spark-testnb \
        -i ${WORK_DIR}/spamassassin-test-vectors\
        -m ${WORK_DIR}/spark-model $c -ma $MASTER

    fi
elif [ "x$alg" == "xsgd" ]; then
  if [ ! -e "/tmp/news-group.model" ]; then
    echo "Training on ${WORK_DIR}/spamassassin-bydate/spamassassin-bydate-train/"
    ./bin/mahout org.apache.mahout.classifier.sgd.TrainNewsGroups ${WORK_DIR}/spamassassin-bydate/spamassassin-bydate-train/
  fi
  echo "Testing on ${WORK_DIR}/spamassassin-bydate/spamassassin-bydate-test/ with model: /tmp/news-group.model"
  ./bin/mahout org.apache.mahout.classifier.sgd.TestNewsGroups --input ${WORK_DIR}/spamassassin-bydate/spamassassin-bydate-test/ --model /tmp/news-group.model
elif [ "x$alg" == "xclean" ]; then
  rm -rf $WORK_DIR
  rm -rf /tmp/news-group.model
  $DFSRM $WORK_DIR
fi
# Remove the work directory
#
