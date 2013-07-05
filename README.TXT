GENERAL INFO:
=============
This classification framework contains a self written Naive Bayes implementation
as current solutions lack the ability of extracting probabilities for certain
features.

The framework furthermore contains a modified version of libSVM implemented by
Chih-Chung Chang and Chih-Jen Lin. I've tried to comment as much as possible
through learning the algorithm as the Java source code of libSVM lacks proper
explanations and is therefore not well-suited for learning the algorithm. I've
also tried to remove as many static function calls and to modularize the code.

For C45 I've modified the fastC45 implementation by Ping He and Xiaohua Xu. 
Loading of UCI data sets needed to be modified and I've extended the code base
to support loading of ARFF files.

ToDo:
=====
*) change framework so that C45, naive Bayes and SVM all can handle the same 
   data. Currently Naive Bayes supports Java Generics for Features and Concepts
   but can not decide yet if the features are continuous, ordinal or nominal.
   SVM only works on continuous data and C45 is able to work on all kind of data
   though it does no work on ordinal data currently.
*) Extend MetaData to accept ordinal data like Dates or nominal values like
   small < normal < tall
*) Storage improvement for large-scale data needed. Current solution requires
   to store the model completely in memory. Naive Bayes only stores the features
   and their counts in memory - but on large scale data like building trigrams
   from parsed HTML tokens and parsing 180.000 HTML pages fill the RAM up to
   9 gigabytes and beyond!
*) map-reduce would be nice as this might speed up the training process (for 
   naive Bayes this should be trivial). In regards to Java 8 this should help
   using multicore computation on a local machine too.