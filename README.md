# CMSC: Collaborative Multi-Domain Sentiment Classification

A matlab implementation of the collaborative multi-domain sentiment classification algorithm.

By Fangzhao Wu, Zhigang Yuan and Yongfeng Huang.

#Introduction
This source code is designed to implement the collaborative multi-domain sentiment classification approach (CMSC) proposed in reference [1]. The aim of CMSC approach is to train train domain-specific sentiment classifiers for multiple domains simultaneously in a collaborative way. The sentiment information in different domains is shared to train more accurate and robust sentiment classifiers for each domain when labeled data is scarce. In CMSC approach, the sentiment classifier of each domain is decomposed into two components, a global one and a domain-specific one. The global model can capture the general sentiment knowledge and is shared by various domains. The domain-specific model can capture the specific sentiment expressions in each domain. CMSC also incorporates the domain-specific sentiment knowledge mined from both labeled and unlabeled samples in each domain to enhance the learning of domain-specific sentiment classifiers. The similarities between different domains are incorporated into CMSC approach as regularization over the domain-specific sentiment classifiers to encourage the sharing of sentiment information between similar domains.


#Citation

If you use this package and like it, welcome to cite this manuscript:

[1] Fangzhao Wu, Zhigang Yuan, and Yongfeng Huang. Collaboratively Training Sentiment Classifiers for Multiple Domains. IEEE Transactions on Knowledge and Data Engineering, under review.

