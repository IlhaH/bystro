def make_pca(X, y, col1=0, col2=1):
    for pop in y.unique():
        mask = y == pop
        plt.scatter(X[mask, col1], X[mask, col2], label=pop)
    plt.legend()
    plt.show()

    def all_vs_all_log_reg_experiment():
    from sklearn.linear_model import LogisticRegression

    results = {}
    C = 10**-2
    penalty = "l1"
    solver = "liblinear"
    for pop1, pop2 in itertools.combinations(POPULATIONS, 2):
        print("starting on:", pop1, pop2)
        train_mask = train_y.population.isin([pop1, pop2])
        test_mask = test_y.population.isin([pop1, pop2])
        train_X_med = train_Xpc[train_mask, :]
        train_y_med = (train_y.population[train_mask] == pop2) * 1
        test_X_med = test_Xpc[test_mask, :]
        test_y_med = (test_y.population[test_mask] == pop2) * 1

        logreg = LogisticRegression(max_iter=1000, C=C, penalty=penalty, solver=solver)
        logreg.fit(train_X_med, train_y_med)
        train_phat = logreg.predict_proba(train_X_med)[:, 1]
        test_phat = logreg.predict_proba(test_X_med)[:, 1]

        train_accuracy = accuracy_score(train_y_med, train_phat > 0.5)
        test_accuracy = accuracy_score(test_y_med, test_phat > 0.5)
        train_roc = roc_auc_score(train_y_med, train_phat)
        test_roc = roc_auc_score(test_y_med, test_phat)

        results[pop1, pop2] = {
            "train_roc": train_roc,
            "test_roc": test_roc,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
    sorted_results = sorted(results.items(), key=lambda kv: kv[1]["test_roc"])
    for (pop1, pop2), (scores) in sorted_results:
        if any(v < 1 for v in scores.values()):
            desc1, desc2 = description_from_pop[pop1], description_from_pop[pop2]
            print(pop1, desc1, "|", pop2, desc2)
            for k, v in scores.items():
                print("\t", k, v)
    metrics = ["train_roc", "test_roc", "train_accuracy", "test_accuracy"]
    for metric in metrics:
        xs = np.array([scores[metric] for scores in results.values()])
        mean_metric = np.mean(xs)
        mean_metric_lt1 = np.mean(xs[xs < 1])
        print("mean", metric, mean_metric, mean_metric_lt1)


def ovr_experiment():
    from sklearn.multiclass import OneVsRestClassifier

    C = 10**-3
    logreg = LogisticRegression(C=C, max_iter=1000)
    ovr = OneVsRestClassifier(logreg)
    ovr.fit(train_Xpc, train_y_pop)
    train_yhat = ovr.predict(train_Xpc)
    test_yhat = ovr.predict(test_Xpc)
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print("OVR log reg experiment")
    print("C", C)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)


def ovr_svm_experiment():
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    C = 10**0
    kernel = "linear"
    svc = SVC(C=C)
    ovr = OneVsRestClassifier(svc)
    ovr.fit(train_Xpc, train_y_pop)
    train_yhat = ovr.predict(train_Xpc)
    test_yhat = ovr.predict(test_Xpc)
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print("OVR SVM experiment")
    print("C", C)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)


def ovo_experiment():
    from sklearn.multiclass import OneVsOneClassifier

    C = 10**-4
    logreg = LogisticRegression(C=C, max_iter=1000)
    ovo = OneVsOneClassifier(logreg)
    ovo.fit(train_Xpc, train_y_pop)
    train_yhat = ovo.predict(train_Xpc)
    test_yhat = ovo.predict(test_Xpc)
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print("OVO log reg experiment")
    print("C", C)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)

    print(train_accuracy)
    print(test_accuracy)


def knn_experiment():
    from sklearn.neighbors import KNeighborsClassifier

    n_neighbors = 20
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_Xpc, train_y_pop)
    train_yhat = knn.predict(train_Xpc)
    test_yhat = knn.predict(test_Xpc)
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print("KNN log reg experiment")
    print("C", C)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)

    print(train_accuracy)
    print(test_accuracy)


def gaussian_nb_experiment(dims=10):
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(train_Xpc[:, :dims], train_y_pop)
    train_yhat = clf.predict(train_Xpc[:, :dims])
    test_yhat = clf.predict(test_Xpc[:, :dims])
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print("GNB experiment")
    print("dims", dims)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)

    print(train_accuracy)
    print(test_accuracy)


def bernoulli_nb_experiment(dims=10_000):
    from sklearn.naive_bayes import BernoulliNB

    title = "bernoulli nb"
    clf = BernoulliNB()
    clf.fit(train_X.to_numpy()[:, :dims], train_y_pop)
    train_yhat = clf.predict(train_X.to_numpy()[:, :dims])
    test_yhat = clf.predict(test_X.to_numpy()[:, :dims])
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print(title)
    print("dims", dims)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)

    print(train_accuracy)
    print(test_accuracy)


def multinomial_nb_experiment(dims=10_000, alpha=1):
    from sklearn.naive_bayes import MultinomialNB

    title = "multinomial nb"
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train_X_filtered.to_numpy()[:, :dims], train_y_pop)
    train_yhat = clf.predict(train_X_filtered.to_numpy()[:, :dims])
    test_yhat = clf.predict(test_X_filtered.to_numpy()[:, :dims])
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print(title)
    print("dims:", dims)
    print("alpha:", alpha)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)

    print(train_accuracy)
    print(test_accuracy)


def gradient_boosting_experiment(
    dims=1_000_000, n_estimators=1000, max_depth=1, min_samples_leaf=1, learning_rate=0.1, verbose=1
):
    from sklearn.ensemble import GradientBoostingClassifier

    title = "gradient boosting"
    clf = GradientBoostingClassifier(
        max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, verbose=verbose
    )
    clf.fit(train_Xpc[:, :dims], train_y_pop)
    train_yhat = clf.predict(train_Xpc[:, :dims])
    test_yhat = clf.predict(test_Xpc[:, :dims])
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print(title)

    print("max depth:", max_depth)
    print("num estimators:", n_estimators)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)

    print(train_accuracy)
    print(test_accuracy)


def get_pca_roc_auc_score(X1, X2, j):
    x1 = X1[:, j]
    x2 = X2[:, j]
    x = np.hstack([x1, x2])
    y = [0 for _ in x1] + [1 for _ in x2]
    return roc_auc_score(y, x)


def multilevel_experiment():
    for superpop in train_y.superpop.unique():
        superpop = "EUR"
        train_pop_mask = train_y.superpop == superpop
        test_pop_mask = test_y.superpop == superpop
        pops = train_y.population[pop_mask].unique()

        train_Xsp = train_X_filtered.loc[train_pop_mask, :]
        test_Xsp = test_X_filtered.loc[test_pop_mask, :]
        train_y_sp = train_y.population[train_pop_mask]
        test_y_sp = test_y.population[test_pop_mask]

        std_mask = train_Xsp.std(axis=0) > 0
        train_Xsp = train_Xsp[train_Xsp.columns[std_mask]]
        test_Xsp = test_Xsp[test_Xsp.columns[std_mask]]

        train_Xsppc, test_Xsppc, pca_sp = _perform_pca(train_Xsp, test_Xsp)
        rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_leaf=5)
        dims = 50
        rfc.fit(train_Xsppc[:, :dims], train_y_sp)
        train_yhat_sp = rfc.predict(train_Xsppc[:, :dims])
        test_yhat_sp = rfc.predict(test_Xsppc[:, :dims])

        print(accuracy_score(train_y_sp, train_yhat_sp))
        print(accuracy_score(test_y_sp, test_yhat_sp))

        print(classification_report(test_y_sp, test_yhat_sp))


def lda_experiment():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_X, train_y_pop)
    train_Xlda = lda.transform(train_X)
    test_Xlda = lda.transform(test_X)

    C = 10**1
    logreg = LogisticRegression(C=C, max_iter=1000)
    ovr = OneVsRestClassifier(logreg)
    ovr.fit(train_Xlda, train_y_pop)
    train_yhat = ovr.predict(train_Xlda)
    test_yhat = ovr.predict(test_Xlda)
    train_accuracy = accuracy_score(train_y_pop, train_yhat)
    # 0.9864479315263909
    test_accuracy = accuracy_score(test_y_pop, test_yhat)
    # 0.8311965811965812
    print("OVR log reg experiment")
    print("C", C)
    print("train accuracy:", train_accuracy)
    print("test accuracy:", test_accuracy)
