#!/usr/bin/env python

from yellowbrick.features import JointPlotVisualizer, Rank2D

from yellowbrick.datasets import load_bikeshare

X, y = load_bikeshare()
print(X.head())

# visualizer = Rank2D(algorithm='pearson')
# visualizer.fit(X, y)                # Fit the data to the visualizer
# visualizer.transform(X)             # Transform the data
# visualizer.show()                   # Finali

visualizer = JointPlotVisualizer(feature='temp', target='feelslike')
visualizer.fit_transform(X['temp'], X['feelslike'])
visualizer.show()