# CMSC
A matlab implementation of the collaborative multi-domain sentiment classification algorithm.
Usage: dsa.py [options] [paramaters]
Options:  -h, --help, display the usage of the DSA commands
          -b, --bigram, if this paramater is set, it means use the unigram and
                bigram features for sentiment classification, otherwise only use the
                unigram features
          -n, --nltk, when this paramater is to set, it means using nltk as the POS tagging
                tool, if not means POS tagging with the stanford-postagger.
          -t, --token path, the token data directory
          -r, --reverse path, the reverse samples directory
          -o, --output path, the directory to save the output files
          -c, --classifier [libsvm|liblinear|nb], the classifier toolkits used for sentiment
                classification, the value 'libsvm', 'liblinear' and 'nb', correspond to libsvm
                classifier, logistic regression classifier and Naive Bayes classifier
                respectively
          -s, --select ratio, the ratio of token samples selected to reverse. If not set, it
                means to reverse all token samples
          -f, --fs_method [CHI|IG|LLR|MI|WLLR] The feature-selecting methods to constructing the
                pseudo-antonym dictionary. If this paramater is not set, it means construct a
                antonym dictionary with wordnet
Paramaters:
       weight conferdence, two paramaters mean essemble a system with 3conf DSA
       weight weight weight weight, four paramaters mean essemble with four system(o2o, o2r, d2o, d2r)
