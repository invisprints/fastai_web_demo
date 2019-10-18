# Starter for deploying [fast.ai](https://www.fast.ai) models on [AWS Elastic Beanstalk](https://aws.amazon.com/cn/elasticbeanstalk/)

This repo can be used as a starting point to deploy [fast.ai](https://github.com/fastai/fastai) models on Elastic Beanstalk.

The project is based on [fasi.ai render-examples](https://github.com/render-examples/fastai-v3) and [Deploying on AWS BeanStalk](https://course.fast.ai/deployment_aws_beanstalk.html)

Notice: In the tutorial they recommended to use t3.small server, but I found in many cases it didn't work. Switch it to t3.medium solve many unknown problems!

The model link in `server.py` refer to a garbage classification, you can check the [notebook](https://www.kaggle.com/invisprints/garbage-classification-via-efficientnet-model) for more information!