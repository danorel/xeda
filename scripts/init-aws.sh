#!/bin/bash

pip install awscli-local==0.21

awslocal s3 mb s3://$AWS_S3_BUCKET_NAME
