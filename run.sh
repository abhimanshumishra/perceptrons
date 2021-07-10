# perceptron without stopword removal
python perceptron.py train test 0.01 100 no >> acc.txt
python perceptron.py train test 0.02 100 no >> acc.txt
python perceptron.py train test 0.03 100 no >> acc.txt
python perceptron.py train test 0.04 100 no >> acc.txt
python perceptron.py train test 0.05 100 no >> acc.txt
python perceptron.py train test 0.1 100 no >> acc.txt
python perceptron.py train test 0.5 100 no >> acc.txt
python perceptron.py train test 0.8 100 no >> acc.txt
python perceptron.py train test 1 100 no >> acc.txt
python perceptron.py train test 0.01 200 no >> acc.txt
python perceptron.py train test 0.02 200 no >> acc.txt
python perceptron.py train test 0.03 200 no >> acc.txt
python perceptron.py train test 0.04 200 no >> acc.txt
python perceptron.py train test 0.05 200 no >> acc.txt
python perceptron.py train test 0.1 200 no >> acc.txt
python perceptron.py train test 0.5 200 no >> acc.txt
python perceptron.py train test 0.8 200 no >> acc.txt
python perceptron.py train test 1 200 no >> acc.txt
python perceptron.py train test 0.01 150 no >> acc.txt
python perceptron.py train test 0.02 150 no >> acc.txt
python perceptron.py train test 0.03 150 no >> acc.txt
python perceptron.py train test 0.04 150 no >> acc.txt
python perceptron.py train test 0.05 150 no >> acc.txt
python perceptron.py train test 0.1 150 no >> acc.txt
python perceptron.py train test 0.5 150 no >> acc.txt
python perceptron.py train test 0.8 150 no >> acc.txt
python perceptron.py train test 1 150 no >> acc.txt

# perceptron with stopword removal
python perceptron.py train test 0.01 100 yes >> acc.txt
python perceptron.py train test 0.02 100 yes >> acc.txt
python perceptron.py train test 0.03 100 yes >> acc.txt
python perceptron.py train test 0.04 100 yes >> acc.txt
python perceptron.py train test 0.05 100 yes >> acc.txt
python perceptron.py train test 0.1 100 yes >> acc.txt
python perceptron.py train test 0.5 100 yes >> acc.txt
python perceptron.py train test 0.8 100 yes >> acc.txt
python perceptron.py train test 1 100 yes >> acc.txt
python perceptron.py train test 0.01 200 yes >> acc.txt
python perceptron.py train test 0.02 200 yes >> acc.txt
python perceptron.py train test 0.03 200 yes >> acc.txt
python perceptron.py train test 0.04 200 yes >> acc.txt
python perceptron.py train test 0.05 200 yes >> acc.txt
python perceptron.py train test 0.1 200 yes >> acc.txt
python perceptron.py train test 0.5 200 yes >> acc.txt
python perceptron.py train test 0.8 200 yes >> acc.txt
python perceptron.py train test 1 200 yes >> acc.txt
python perceptron.py train test 0.01 150 yes >> acc.txt
python perceptron.py train test 0.02 150 yes >> acc.txt
python perceptron.py train test 0.03 150 yes >> acc.txt
python perceptron.py train test 0.04 150 yes >> acc.txt
python perceptron.py train test 0.05 150 yes >> acc.txt
python perceptron.py train test 0.1 150 yes >> acc.txt
python perceptron.py train test 0.5 150 yes >> acc.txt
python perceptron.py train test 0.8 150 yes >> acc.txt
python perceptron.py train test 1 150 yes >> acc.txt

# naive bayes
python naive_bayes.py train test yes yes >> acc.txt
python naive_bayes.py train test yes no >> acc.txt
python naive_bayes.py train test no no >> acc.txt
python naive_bayes.py train test no yes >> acc.txt

echo "\n\n" >> acc.txt
