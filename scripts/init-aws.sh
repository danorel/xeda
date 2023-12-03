#!/bin/bash

pip install awscli-local==0.21

awslocal s3 mb s3://$AWS_STORAGE_BUCKET_NAME
