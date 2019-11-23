for VARIABLE in 'Classical_models/baseline_train.py' 'deep_learning_models/lstm_train.py'

do
	python3 $VARIABLE && kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')
done
