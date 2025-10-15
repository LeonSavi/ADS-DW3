
def objective_bagTree(trial, data_: list[pd.DataFrame]) -> float:
    X_train, X_val, y_train, y_val = data_

    dt_max_depth         = trial.suggest_int("dt_max_depth", 8, 14)
    dt_min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 6)
    dt_min_samples_leaf  = trial.suggest_int("dt_min_samples_leaf", 6, 10)
    dt_max_features      = trial.suggest_categorical("dt_max_features", [None, "sqrt", "log2", 0.6, 0.8, 1.0])

    use_mid = trial.suggest_categorical("dt_use_min_impurity_decrease", [False, True])
    dt_min_impurity_decrease = 0.0 if not use_mid else trial.suggest_float("dt_min_impurity_decrease_pos", 1e-6, 1e-2)
    dt_ccp_alpha = trial.suggest_float("dt_ccp_alpha", 0.0, 0.004)  # small for bagging friendliness

    base_dt = DecisionTreeRegressor(
        max_depth=dt_max_depth,
        min_samples_split=dt_min_samples_split,
        min_samples_leaf=dt_min_samples_leaf,
        max_features=dt_max_features,
        min_impurity_decrease=dt_min_impurity_decrease,
        ccp_alpha=dt_ccp_alpha,
        criterion="squared_error",
        splitter="best",
        random_state=1,
    )

    use_bagging = trial.set_user_attr("use_bagging", True)

    # use_bagging = trial.suggest_categorical("use_bagging", [True,True])

    if use_bagging:
        bag_n_estimators = trial.suggest_int("bag_n_estimators", 20, 400, step=20)
        bag_max_samples  = trial.suggest_float("bag_max_samples", 0.5, 1.0)
        bag_max_features = trial.suggest_float("bag_max_features", 0.5, 1.0)
        bag_bootstrap    = trial.suggest_categorical("bag_bootstrap", [True, False])
        bag_bootstrap_f  = trial.suggest_categorical("bag_bootstrap_features", [False, True])

        model = BaggingRegressor(
            estimator=base_dt,
            n_estimators=bag_n_estimators,
            max_samples=bag_max_samples,
            max_features=bag_max_features,
            bootstrap=bag_bootstrap,
            bootstrap_features=bag_bootstrap_f,
            oob_score=False,# we already have a val set
            n_jobs=-1,
            random_state=1,
            verbose=0,
        )
        trial.set_user_attr("model_type", "Bagging(DecisionTree)")
    else:
        model = base_dt
        trial.set_user_attr("model_type", "DecisionTree")

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, pred)

    trial.set_user_attr("rmse", rmse)

    trial_metrics(trial,model,data_)
    
    return rmse




def objective_gbr(trial, data_: list[pd.DataFrame]) -> float:
    """
    Optuna objective for scikit-learn GradientBoostingRegressor.
    Expects data_ = [X_train, X_val, y_train, y_val].
    Returns RMSE (to mirror your XGB objective).
    """
    X_train, X_val, y_train, y_val = data_

    loss_ = trial.suggest_categorical("gbr_loss", ["squared_error", "huber", "absolute_error"])

    if loss_ == "huber":
        alpha_ = trial.suggest_float("gbr_alpha", 0.85, 0.99)
    else:
        alpha_ = 0.9

    criterion_ = trial.suggest_categorical("gbr_criterion", ["friedman_mse", "squared_error"])

    n_estimators_ = trial.suggest_int("gbr_n_estimators", 100, 1000, step=20)
    lr_          = trial.suggest_float("gbr_learning_rate", 0.01, 0.3, log=False)
    max_depth_   = trial.suggest_int("gbr_max_depth", 2, 6)                 
    subsample_   = trial.suggest_float("gbr_subsample", 0.6, 1.0)           
    max_features_= trial.suggest_categorical("gbr_max_features",
                                             [None, "sqrt", "log2", 0.5, 0.7, 1.0])
    min_samples_split_ = trial.suggest_int("gbr_min_samples_split", 2, 30)
    min_samples_leaf_  = trial.suggest_int("gbr_min_samples_leaf", 1, 15)

    use_max_leaf_nodes_ = trial.suggest_categorical("gbr_use_max_leaf_nodes", [False, True])
    max_leaf_nodes_ = trial.suggest_int("gbr_max_leaf_nodes", 16, 256) if use_max_leaf_nodes_ else None

    use_mid_ = trial.suggest_categorical("gbr_use_min_impurity_decrease", [False, True])
    min_impurity_decrease_ = (
        0.0 if not use_mid_
        else trial.suggest_float("gbr_min_impurity_decrease_pos", 1e-6, 0.01, log=False)
    )

    use_ccp_ = trial.suggest_categorical("gbr_use_ccp_alpha", [False, True])
    ccp_alpha_ = (
        0.0 if not use_ccp_
        else trial.suggest_float("gbr_ccp_alpha_pos", 1e-6, 0.01, log=False)
    )

    n_iter_no_change_ = trial.suggest_int("gbr_n_iter_no_change", 5, 40)
    val_fraction_     = trial.suggest_float("gbr_validation_fraction", 0.1, 0.4)
    tol_              = trial.suggest_float("gbr_tol", 1e-6, 1e-3, log=False)


    kf_rmses = []
    for i, (x_train, x_test, y_train, y_test) in enumerate(zip(*data_)):
        model = GradientBoostingRegressor(
            loss=loss_,
            learning_rate=lr_,
            n_estimators=n_estimators_,
            subsample=subsample_,
            criterion=criterion_,
            min_samples_split=min_samples_split_,
            min_samples_leaf=min_samples_leaf_,
            min_impurity_decrease=min_impurity_decrease_,
            max_depth=max_depth_,
            max_features=max_features_,
            max_leaf_nodes=max_leaf_nodes_,
            ccp_alpha=ccp_alpha_,
            validation_fraction=val_fraction_,
            n_iter_no_change=n_iter_no_change_,
            tol=tol_,
            random_state=1,
            verbose=0,
        )

        if loss_ == "huber":
            model.set_params(alpha=alpha_)

        model.fit(x_train, y_train)

        kf_data = [x_train, x_test, y_train, y_test]
        rmse_k = trial_metrics(trial, model, kf_data, fold_idx=i)
        kf_rmses.append(rmse_k)

    mean_rmse = float(np.mean(kf_rmses))
    std_rmse = float(np.std(kf_rmses))
    trial.set_user_attr("kf_rmse_mean", mean_rmse)
    trial.set_user_attr("kf_rmse_std", std_rmse)

    return mean_rmse


def objective_nn(trial, data_:list):
    """Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of filters of 

    """

    x_train, x_test, y_train, y_test = data_


    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers_ = [trial.suggest_int(f"l{i}_nodes", 1, 128, step=1) for i in range(n_layers)]
    activation_ = [trial.suggest_categorical(f"l{i}_activation", ["relu", "gelu", "leakyrelu", "softplus"]) for i in range(n_layers)]
    lr_ = trial.suggest_float('learning_rate', 1e-4,1e-3,log=True)
    batch_size_ = trial.suggest_int('batch_size', 2, 128, step=1)
    n_epochs = trial.suggest_int('n_epochs', 50, 500, step = 50)

    # Generate the model
    NN_model = Model(n_features=len(x_train.columns),
                     layers=layers_,
                     activations=activation_,
                     dropout=dropout_prob)



    trainer = Trainer(model = NN_model,
                      data = data_,
                      batch_size = batch_size_,
                      lr = lr_,
                      epochs=n_epochs)

    return trainer.evaluate()