# Korean_ABSA

Aspect-Based Sentiment Analysis  
Language: Korean  
Dataset: https://corpus.korean.go.kr/task/taskDownload.do?taskId=8&clCd=ING_TASK&subMenuId=sub02  
Baseline: https://github.com/teddysum/korean_ABSA_baseline

## How to use
### Train
`$ mlflow ui`
`$ bash train.sh {MLflow Experiment Name} {MLflow Run Name}`

### Test
`$ bash test.sh`
<br><br>
## To Do
- early stopping
- get best model automatically from training
- research other models
- freeze requirements
