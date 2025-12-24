import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def initialize(data, k):
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0])]) # First centroid is random 

    for _ in range(k - 1):
        distances = []
        for point in data:
            min_dist = min([distance(point, c) for c in centroids]) # Distance to closest centroid
            distances.append(min_dist)
        
        next_centroid = data[np.argmax(distances)] # farthest point is the next centroid
        centroids.append(next_centroid)
    
    return np.array(centroids)

def initialize(data, k):
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0])]) # First centroid is random 

    for _ in range(k - 1):
        distances = []
        for point in data:
            min_dist = min([distance(point, c) for c in centroids]) # Distance to closest centroid
            distances.append(min_dist)
        
        next_centroid = data[np.argmax(distances)] # farthest point is the next centroid
        centroids.append(next_centroid)
    
    return np.array(centroids)
def assign_to_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = []
        for c in centroids:
            distances.append(distance(point, c))

        cluster = np.argmin(distances) # index of closest centroid
        clusters.append(cluster) # assign point to closest centroid
    return np.array(clusters) # array of cluster for every point

def assign_to_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = []
        for c in centroids:
            distances.append(distance(point, c))

        cluster = np.argmin(distances) # index of closest centroid
        clusters.append(cluster) # assign point to closest centroid
    return np.array(clusters) # array of cluster for every point

def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i] # get points assigned to cluster i
        new_centroid = np.mean(cluster_points, axis=0) # mean of points in cluster i
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def calculate_inertia(data, centroids, clusters):
    inertia = 0
    for i, point in enumerate(data):
        centroid = centroids[clusters[i]] # centroid of assigned cluster
        inertia += np.linalg.norm(point - centroid)**2 # sum of squared norm of point to centroid
    return inertia

def kmeans(data, centroids, tolerance, max_iterations):
    inertia_history = []
    startTime = time.time()
    while(max_iterations > 0):
        clusters = assign_to_clusters(data, centroids)
        inertia = calculate_inertia(data, centroids, clusters)
        new_centroids = update_centroids(data, clusters, len(centroids))
        inertia_history.append(inertia) 
        diff=0
        for i in range(len(centroids)):
            diff += distance(centroids[i], new_centroids[i])
        centroids = new_centroids

        if diff < tolerance:
            break
        max_iterations -= 1
    endTime = time.time()
    convergenceTime = endTime - startTime
    return centroids, clusters, inertia_history, convergenceTime