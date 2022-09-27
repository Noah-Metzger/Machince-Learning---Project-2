class KNN:
    def __init__(self, df, test, test_y, train, train_y, k):
        self.df = df
        self.test = test
        self.test_y = test_y
        self.train = train
        self.train_y = train_y
        self.k = k

    def lpNorm(self, x, y, p):
        """

        :param x:
        :param y:
        :param p:
        :return:
        """
        sums = 0
        for i in range(len(x)):
            sums += pow(x[i] - y[i], p)
        return pow(sums, (1 / p))

    def kernel(self, x, y, sigma):
        """

        :param x:
        :param y:
        :param sigma:
        :return:
        """
        r = 1 / (2 * sigma)
        return math.exp(-r * lpNorm(x, y, 2))

    def predictInstance(self, testInstance):
        neighbors = []
        classes = []
        indices = []
        for i, row in train.iterrows():
            dist = lpNorm(np.array(row), np.array(testInstance), 2)
            neighbors.append(dist)
            classes.append(train_y[i])
            indices.append(i)

        for i in range(len(neighbors)):
            for j in range(0, len(neighbors) - i - 1):
                if neighbors[j] > neighbors[j + 1]:
                    neighbors[j], neighbors[j + 1] = neighbors[j + 1], neighbors[j]
                    classes[j], classes[j + 1] = classes[j + 1], classes[j]
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]

        nearestNeighbors = classes[:k]
        nearestIndices = indices[:k]

        if isClassification:
            vote = np.unique(nearestNeighbors)
            count = []
            for i in range(len(vote)):
                count.append(0)

            for i in nearestNeighbors:
                for j, cl in enumerate(vote):
                    if i == cl:
                        count[j] += 1
            return vote[np.argmax(count)]
        else:
            nom = 0
            dom = 0
            for i in nearestIndices:
                kern = kernel(train.loc[i], testInstance, bandwidth)
                nom += kern * train_y[i]
                dom += kern

            return (nom / dom)

    def kknRegular(self, test, test_y, train, train_y, k, isClassification, bandwidth, error):
        """

        :param test:
        :param train:
        :param train_y:
        :param k:
        :param isClassification:
        :return:
        """
        predictions = []
        for ind, testInstance in test.iterrows():
            neighbors = []
            classes = []
            indices = []
            for i, row in train.iterrows():
                dist = lpNorm(np.array(row), np.array(testInstance), 2)
                neighbors.append(dist)
                classes.append(train_y[i])
                indices.append(i)

            for i in range(len(neighbors)):
                for j in range(0, len(neighbors) - i - 1):
                    if neighbors[j] > neighbors[j + 1]:
                        neighbors[j], neighbors[j + 1] = neighbors[j + 1], neighbors[j]
                        classes[j], classes[j + 1] = classes[j + 1], classes[j]
                        indices[j], indices[j + 1] = indices[j + 1], indices[j]

            nearestNeighbors = classes[:k]
            nearestIndices = indices[:k]

            if isClassification:
                vote = np.unique(nearestNeighbors)
                count = []
                for i in range(len(vote)):
                    count.append(0)

                for i in nearestNeighbors:
                    for j, cl in enumerate(vote):
                        if i == cl:
                            count[j] += 1
                predictions.append(vote[np.argmax(count)])
            else:
                nom = 0
                dom = 0
                for i in nearestIndices:
                    kern = kernel(train.loc[i], testInstance, bandwidth)
                    nom += kern * train_y[i]
                    dom += kern

                predictions.append(nom / dom)
        return predictions

    def kknEdited(self, df):
        for i, row in df.iterrows:
