import keras.backend as K


def ratio(target, y_true, y_pred):
    """
    ratio of 1 category
    use it this way:
    def mp_ratio(y_true,y_pred):
        return ratio(1,y_true,y_pred)
    """
    return K.mean(K.equal(K.argmax(y_pred, axis=-1), target))


def precision(target, y_true, y_pred):
    """
    Precision:
        Precision is the fraction of detections
        reported by the model that were correct.
    target: the target category

    Use it as:
    def mp_prec(y_true, y_pred):
        return precision(1, y_true, y_pred)
    """
    y_true_arg = K.argmax(y_true, axis=-1)
    y_pred_arg = K.argmax(y_pred, axis=-1)
    # Ground truthly, how many data belongs to the target category
    istarg = K.cast(K.equal(y_true_arg, target), K.floatx())
    # The distribution of "Model guessed right"
    isright = K.cast(K.equal(y_true_arg, y_pred_arg), K.floatx())
    # The distribution of model predicted as target cate, right or wrong
    preistarg = K.cast(K.equal(y_pred_arg, target), K.floatx())
    return K.mean(K.equal(istarg * isright,preistarg))
    # return K.cast(K.sum(istarg * isright) / (K.sum(preistarg)), K.floatx())

def recall(target, y_true, y_pred):
    """
    Recall:
        Recall is the fraction of true events that were detected
    target: the target category
    Use it as:
    def mp_rcal(y_true, y_pred):
        return recall(1, y_true, y_pred)
    """
    y_true_arg = K.argmax(y_true, axis=-1)
    y_pred_arg = K.argmax(y_pred, axis=-1)
    # Ground truth, how many data belongs to the target category
    istarg = K.cast(K.equal(y_true_arg, target), K.floatx())
    # The distribution of "Model guessed right"
    isright = K.cast(K.equal(y_true_arg, y_pred_arg), K.floatx())
    return K.mean(K.equal(istarg * isright,istarg))

def fscore(target, y_true, y_pred):
    """
    fscore:
    Use it as:
    def mp_f(y_true, y_pred):
        return fscore(1, y_true, y_pred)
    """
    rc = recall(target, y_true, y_pred)
    prec = precision(target, y_true, y_pred)
    return 2*(rc*prec)/(rc+prec)