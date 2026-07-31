# Machine Learning Basics

A project-authored fixture book. The text is original and deliberately short; it
exists so ingestion, citation and planning can be verified without shipping
copyrighted material.

## Supervised Learning

Supervised learning fits a function from labelled examples so that it predicts
the label of examples it has not seen. The fit is judged on held-out data, never
on the data it was trained on, because a model can memorise its training set
without learning anything transferable.

Classification predicts a discrete label; regression predicts a continuous
value. The distinction changes the loss function and the evaluation metric, but
not the shape of the training loop.

## Overfitting and Regularisation

A model overfits when it captures noise specific to the training sample. The
symptom is a widening gap between training error and validation error as
capacity grows: training error keeps falling while validation error turns
upward.

Regularisation constrains capacity to close that gap. An L2 penalty shrinks
coefficients toward zero; dropout removes units at random during training; early
stopping halts before the validation curve turns. All three trade a little bias
for a large reduction in variance. Supervised learning is assumed knowledge
here, since overfitting is defined against a training and validation split.

## Gradient Descent

Gradient descent minimises a loss by stepping in the direction of steepest
descent, scaled by a learning rate. Too large a rate diverges; too small a rate
converges so slowly that training never finishes.

Stochastic gradient descent estimates the gradient from a minibatch rather than
the full dataset, which makes each step cheap and noisy. The noise is useful: it
helps the optimiser escape shallow local minima that full-batch descent would
settle into.

## Evaluation Metrics

Accuracy is misleading on imbalanced data: a classifier that always predicts the
majority class scores well and is useless. Precision and recall separate the two
failure modes — precision measures how many predicted positives were right,
recall how many actual positives were found.

The F1 score is their harmonic mean, chosen over the arithmetic mean because it
punishes a model that sacrifices one for the other. Which metric matters is a
property of the application, not of the model.

## Feature Engineering

Features determine what a model is able to learn. Numerical features are usually
scaled so that no single feature dominates a distance or a gradient by virtue of
its units; categorical features are encoded, one-hot for low cardinality and
target or hash encoding for high cardinality.

Hashing a high-cardinality categorical feature into a fixed number of buckets is
the same collision trade-off a hash table makes: it bounds memory at the cost of
occasionally mapping two distinct categories to the same slot.
