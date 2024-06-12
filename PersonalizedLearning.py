import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class PersonalizedLearning:
    def __init__(self, student_data, course_materials):
        self.student_data = pd.DataFrame(student_data)
        self.course_materials = course_materials
        self.model = None
        self.student_clusters = None
        self.validate_data()

    def validate_data(self):
        if self.student_data.isnull().values.any():
            raise ValueError("Les données des étudiants contiennent des valeurs manquantes.")
        for material in self.course_materials:
            if 'subject' not in material or 'difficulty' not in material:
                raise ValueError("Chaque matériel de cours doit avoir un sujet et une difficulté.")

    def preprocess_data(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.student_data.drop('StudentID', axis=1))
        return scaled_data

    def build_autoencoder(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(64, activation="relu")(input_layer)
        encoder = Dense(32, activation="relu")(encoder)
        encoder = Dense(10, activation="relu")(encoder)
        decoder = Dense(32, activation="relu")(encoder)
        decoder = Dense(64, activation="relu")(decoder)
        decoder = Dense(input_dim, activation="sigmoid")(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def cluster_students(self, num_clusters=3, method='gmm'):
        scaled_data = self.preprocess_data()
        if method == 'gmm':
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
            self.student_clusters = gmm.fit_predict(scaled_data)
        elif method == 'hierarchical':
            Z = linkage(scaled_data, method='ward')
            self.student_clusters = fcluster(Z, num_clusters, criterion='maxclust') - 1
        elif method == 'neural':
            autoencoder = self.build_autoencoder(scaled_data.shape[1])
            autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=8, shuffle=True)
            encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)
            encoded_data = encoder.predict(scaled_data)
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
            self.student_clusters = gmm.fit_predict(encoded_data)
        self.student_data['Cluster'] = self.student_clusters

    def assign_materials(self):
        self.cluster_students()
        cluster_materials = {i: [] for i in range(max(self.student_clusters) + 1)}
        for cluster in cluster_materials.keys():
            cluster_students = self.student_data[self.student_data['Cluster'] == cluster]
            cluster_mean_scores = cluster_students.mean(axis=0)
            suitable_materials = [material for material in self.course_materials 
                                  if self.is_material_suitable(material, cluster_mean_scores)]
            cluster_materials[cluster] = suitable_materials
        return cluster_materials

    def is_material_suitable(self, material, cluster_mean_scores):
        difficulty = material.get('difficulty', 0)
        student_skill = cluster_mean_scores.get(material.get('subject', ''), 0)
        return student_skill >= difficulty

    def evaluate(self, student_id, completed_materials):
        if len(completed_materials) == 0:
            return 0
        
        student_cluster = self.student_data[self.student_data['StudentID'] == student_id]['Cluster'].values[0]
        cluster_materials = self.assign_materials()[student_cluster]
        correct_answers = 0
        for material in completed_materials:
            if material in [m['title'] for m in cluster_materials]:
                correct_answers += 1
            else:
                print(f"Matériel '{material}' non attribué à ce cluster.")
        accuracy = correct_answers / len(completed_materials)
        return accuracy

    def display_students_by_cluster(self):
        cluster_groups = self.student_data.groupby('Cluster')
        for cluster, students in cluster_groups:
            print(f"\nGroupe {cluster} :")
            for _, student in students.iterrows():
                print(f"  Étudiant ID: {student['StudentID']}, Scores: {student.drop(['StudentID', 'Cluster']).to_dict()}")
                cluster_mean = self.student_data[self.student_data['Cluster'] == cluster].mean(axis=0)
                print(f"  Raison d'appartenance au groupe: Scores proches du centroïde {cluster} (Moyennes: {cluster_mean.drop(['Cluster'])})")

    def adjust_difficulty(self, feedbacks):
        for feedback in feedbacks:
            material_title = feedback['title']
            perceived_difficulty = feedback['perceived_difficulty']
            for material in self.course_materials:
                if material['title'] == material_title:
                    material['difficulty'] = (material['difficulty'] + perceived_difficulty) / 2

    def analyze_feedback(self, feedback_texts):
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC()),
        ])
        feedback_data = [(f['title'], f['feedback']) for f in feedback_texts]
        titles, texts = zip(*feedback_data)
        text_clf.fit(texts, titles)
        return text_clf.predict(texts)

    def recommend_materials(self, student_id, num_recommendations=3):
        scaled_data = self.preprocess_data()
        student_index = self.student_data[self.student_data['StudentID'] == student_id].index[0]
        student_data = scaled_data[student_index].reshape(1, -1)
        knn = NearestNeighbors(n_neighbors=num_recommendations + 1)
        knn.fit(scaled_data)
        distances, indices = knn.kneighbors(student_data)
        recommendations = []
        for idx in indices[0][1:]:
            cluster = self.student_data.iloc[idx]['Cluster']
            recommendations.append(self.assign_materials()[cluster])
        return recommendations

# Données des étudiants enrichies
students = [
    {'StudentID': 1, 'Math': 85, 'Science': 78, 'English': 92, 'History': 74, 'ResponseTime': 30, 'PlatformInteractions': 100},
    {'StudentID': 2, 'Math': 59, 'Science': 63, 'English': 70, 'History': 58, 'ResponseTime': 45, 'PlatformInteractions': 80},
    {'StudentID': 3, 'Math': 91, 'Science': 85, 'English': 88, 'History': 82, 'ResponseTime': 25, 'PlatformInteractions': 120},
    {'StudentID': 4, 'Math': 52, 'Science': 54, 'English': 65, 'History': 50, 'ResponseTime': 50, 'PlatformInteractions': 60},
    {'StudentID': 5, 'Math': 75, 'Science': 72, 'English': 80, 'History': 68, 'ResponseTime': 35, 'PlatformInteractions': 90},
    {'StudentID': 6, 'Math': 88, 'Science': 90, 'English': 86, 'History': 79, 'ResponseTime': 20, 'PlatformInteractions': 110},
    {'StudentID': 7, 'Math': 55, 'Science': 61, 'English': 60, 'History': 57, 'ResponseTime': 48, 'PlatformInteractions': 70},
    {'StudentID': 8, 'Math': 72, 'Science': 80, 'English': 75, 'History': 70, 'ResponseTime': 33, 'PlatformInteractions': 95},
    {'StudentID': 9, 'Math': 64, 'Science': 66, 'English': 77, 'History': 60, 'ResponseTime': 42, 'PlatformInteractions': 85},
    {'StudentID': 10, 'Math': 95, 'Science': 89, 'English': 84, 'History': 91, 'ResponseTime': 22, 'PlatformInteractions': 130},
    {'StudentID': 11, 'Math': 68, 'Science': 73, 'English': 81, 'History': 67, 'ResponseTime': 34, 'PlatformInteractions': 96},
    {'StudentID': 12, 'Math': 58, 'Science': 60, 'English': 64, 'History': 55, 'ResponseTime': 49, 'PlatformInteractions': 75},
    {'StudentID': 13, 'Math': 93, 'Science': 87, 'English': 90, 'History': 85, 'ResponseTime': 24, 'PlatformInteractions': 125},
    {'StudentID': 14, 'Math': 47, 'Science': 50, 'English': 58, 'History': 45, 'ResponseTime': 52, 'PlatformInteractions': 65},
    {'StudentID': 15, 'Math': 76, 'Science': 70, 'English': 78, 'History': 69, 'ResponseTime': 38, 'PlatformInteractions': 88},
    {'StudentID': 16, 'Math': 89, 'Science': 91, 'English': 85, 'History': 80, 'ResponseTime': 21, 'PlatformInteractions': 115},
    {'StudentID': 17, 'Math': 54, 'Science': 62, 'English': 63, 'History': 56, 'ResponseTime': 47, 'PlatformInteractions': 72},
    {'StudentID': 18, 'Math': 71, 'Science': 77, 'English': 74, 'History': 68, 'ResponseTime': 31, 'PlatformInteractions': 98},
    {'StudentID': 19, 'Math': 63, 'Science': 65, 'English': 76, 'History': 59, 'ResponseTime': 41, 'PlatformInteractions': 83},
    {'StudentID': 20, 'Math': 96, 'Science': 88, 'English': 89, 'History': 92, 'ResponseTime': 23, 'PlatformInteractions': 135},
    {'StudentID': 21, 'Math': 67, 'Science': 74, 'English': 79, 'History': 66, 'ResponseTime': 36, 'PlatformInteractions': 92},
    {'StudentID': 22, 'Math': 57, 'Science': 59, 'English': 63, 'History': 54, 'ResponseTime': 50, 'PlatformInteractions': 78},
    {'StudentID': 23, 'Math': 92, 'Science': 86, 'English': 91, 'History': 84, 'ResponseTime': 26, 'PlatformInteractions': 122},
    {'StudentID': 24, 'Math': 46, 'Science': 51, 'English': 57, 'History': 44, 'ResponseTime': 53, 'PlatformInteractions': 62},
    {'StudentID': 25, 'Math': 77, 'Science': 71, 'English': 79, 'History': 70, 'ResponseTime': 37, 'PlatformInteractions': 90},
    {'StudentID': 26, 'Math': 90, 'Science': 92, 'English': 87, 'History': 81, 'ResponseTime': 22, 'PlatformInteractions': 112},
    {'StudentID': 27, 'Math': 53, 'Science': 63, 'English': 64, 'History': 55, 'ResponseTime': 46, 'PlatformInteractions': 74},
    {'StudentID': 28, 'Math': 70, 'Science': 78, 'English': 73, 'History': 67, 'ResponseTime': 32, 'PlatformInteractions': 97},
    {'StudentID': 29, 'Math': 62, 'Science': 64, 'English': 75, 'History': 58, 'ResponseTime': 40, 'PlatformInteractions': 82},
    {'StudentID': 30, 'Math': 97, 'Science': 89, 'English': 90, 'History': 93, 'ResponseTime': 21, 'PlatformInteractions': 140},
   {'StudentID': 31, 'Math': 66, 'Science': 72, 'English': 80, 'History': 65, 'ResponseTime': 35, 'PlatformInteractions': 94},
    {'StudentID': 32, 'Math': 56, 'Science': 61, 'English': 62, 'History': 53, 'ResponseTime': 48, 'PlatformInteractions': 77},
    {'StudentID': 33, 'Math': 91, 'Science': 85, 'English': 89, 'History': 83, 'ResponseTime': 25, 'PlatformInteractions': 128},
    {'StudentID': 34, 'Math': 45, 'Science': 52, 'English': 59, 'History': 46, 'ResponseTime': 54, 'PlatformInteractions': 66},
    {'StudentID': 35, 'Math': 78, 'Science': 75, 'English': 81, 'History': 72, 'ResponseTime': 36, 'PlatformInteractions': 92},
    {'StudentID': 36, 'Math': 87, 'Science': 89, 'English': 84, 'History': 78, 'ResponseTime': 22, 'PlatformInteractions': 118},
    {'StudentID': 37, 'Math': 52, 'Science': 60, 'English': 65, 'History': 55, 'ResponseTime': 45, 'PlatformInteractions': 71},
    {'StudentID': 38, 'Math': 73, 'Science': 79, 'English': 77, 'History': 69, 'ResponseTime': 30, 'PlatformInteractions': 100},
    {'StudentID': 39, 'Math': 61, 'Science': 66, 'English': 74, 'History': 57, 'ResponseTime': 42, 'PlatformInteractions': 84},
    {'StudentID': 40, 'Math': 98, 'Science': 90, 'English': 92, 'History': 94, 'ResponseTime': 20, 'PlatformInteractions': 138},
    {'StudentID': 41, 'Math': 65, 'Science': 70, 'English': 79, 'History': 63, 'ResponseTime': 37, 'PlatformInteractions': 89},
    {'StudentID': 42, 'Math': 55, 'Science': 58, 'English': 61, 'History': 52, 'ResponseTime': 49, 'PlatformInteractions': 76},
    {'StudentID': 43, 'Math': 90, 'Science': 84, 'English': 88, 'History': 82, 'ResponseTime': 27, 'PlatformInteractions': 120},
    {'StudentID': 44, 'Math': 44, 'Science': 53, 'English': 56, 'History': 45, 'ResponseTime': 55, 'PlatformInteractions': 63},
    {'StudentID': 45, 'Math': 79, 'Science': 76, 'English': 80, 'History': 73, 'ResponseTime': 34, 'PlatformInteractions': 91},
    {'StudentID': 46, 'Math': 86, 'Science': 88, 'English': 83, 'History': 77, 'ResponseTime': 23, 'PlatformInteractions': 113},
    {'StudentID': 47, 'Math': 51, 'Science': 62, 'English': 64, 'History': 54, 'ResponseTime': 46, 'PlatformInteractions': 73},
    {'StudentID': 48, 'Math': 72, 'Science': 78, 'English': 76, 'History': 66, 'ResponseTime': 33, 'PlatformInteractions': 99},
    {'StudentID': 49, 'Math': 60, 'Science': 63, 'English': 73, 'History': 56, 'ResponseTime': 41, 'PlatformInteractions': 81},
    {'StudentID': 50, 'Math': 99, 'Science': 91, 'English': 91, 'History': 95, 'ResponseTime': 19, 'PlatformInteractions': 142},
    {'StudentID': 51, 'Math': 68, 'Science': 74, 'English': 82, 'History': 64, 'ResponseTime': 36, 'PlatformInteractions': 95},
    {'StudentID': 52, 'Math': 54, 'Science': 59, 'English': 62, 'History': 51, 'ResponseTime': 50, 'PlatformInteractions': 78},
    {'StudentID': 53, 'Math': 89, 'Science': 83, 'English': 87, 'History': 81, 'ResponseTime': 26, 'PlatformInteractions': 124},
    {'StudentID': 54, 'Math': 43, 'Science': 50, 'English': 55, 'History': 44, 'ResponseTime': 56, 'PlatformInteractions': 61},
    {'StudentID': 55, 'Math': 80, 'Science': 77, 'English': 82, 'History': 74, 'ResponseTime': 31, 'PlatformInteractions': 90},
    {'StudentID': 56, 'Math': 85, 'Science': 87, 'English': 85, 'History': 76, 'ResponseTime': 24, 'PlatformInteractions': 110},
    {'StudentID': 57, 'Math': 50, 'Science': 61, 'English': 63, 'History': 53, 'ResponseTime': 44, 'PlatformInteractions': 70},
    {'StudentID': 58, 'Math': 75, 'Science': 79, 'English': 78, 'History': 68, 'ResponseTime': 32, 'PlatformInteractions': 97},
    {'StudentID': 59, 'Math': 59, 'Science': 64, 'English': 75, 'History': 55, 'ResponseTime': 40, 'PlatformInteractions': 85},
    {'StudentID': 60, 'Math': 94, 'Science': 89, 'English': 90, 'History': 91, 'ResponseTime': 21, 'PlatformInteractions': 136},
]

# Matériaux de cours
course_materials = [
    {'title': 'Advanced Algebra', 'subject': 'Math', 'difficulty': 75},
    {'title': 'Basic Physics', 'subject': 'Science', 'difficulty': 60},
    {'title': 'Shakespeare', 'subject': 'English', 'difficulty': 80},
    {'title': 'Geometry', 'subject': 'Math', 'difficulty': 65},
    {'title': 'Chemistry', 'subject': 'Science', 'difficulty': 70},
    {'title': 'World History', 'subject': 'History', 'difficulty': 65},
    {'title': 'Modern Literature', 'subject': 'English', 'difficulty': 70},
    {'title': 'Trigonometry', 'subject': 'Math', 'difficulty': 80},
    {'title': 'Biology', 'subject': 'Science', 'difficulty': 75},
    {'title': 'American History', 'subject': 'History', 'difficulty': 70},
    {'title': 'Calculus', 'subject': 'Math', 'difficulty': 85},
    {'title': 'Physics II', 'subject': 'Science', 'difficulty': 80},
    {'title': 'Literary Analysis', 'subject': 'English', 'difficulty': 85},
    {'title': 'European History', 'subject': 'History', 'difficulty': 75},
    {'title': 'Advanced Chemistry', 'subject': 'Science', 'difficulty': 85},
       {'title': 'Introduction to Algebra', 'subject': 'Math', 'difficulty': 50},
    {'title': 'Fundamentals of Biology', 'subject': 'Science', 'difficulty': 55},
    {'title': 'English Grammar Basics', 'subject': 'English', 'difficulty': 40},
    {'title': 'Ancient History', 'subject': 'History', 'difficulty': 60},
    {'title': 'Intermediate Geometry', 'subject': 'Math', 'difficulty': 70},
    {'title': 'Organic Chemistry', 'subject': 'Science', 'difficulty': 80},
    {'title': 'Victorian Literature', 'subject': 'English', 'difficulty': 75},
    {'title': 'Statistics', 'subject': 'Math', 'difficulty': 85},
    {'title': 'Environmental Science', 'subject': 'Science', 'difficulty': 65},
    {'title': 'Asian History', 'subject': 'History', 'difficulty': 70},
    {'title': 'Differential Equations', 'subject': 'Math', 'difficulty': 90},
    {'title': 'Astronomy', 'subject': 'Science', 'difficulty': 70},
    {'title': 'Creative Writing', 'subject': 'English', 'difficulty': 60},
    {'title': 'Middle Ages History', 'subject': 'History', 'difficulty': 75},
    {'title': 'Thermodynamics', 'subject': 'Science', 'difficulty': 85},
    {'title': 'Modern Poetry', 'subject': 'English', 'difficulty': 65},
    {'title': 'Discrete Mathematics', 'subject': 'Math', 'difficulty': 80},
    {'title': 'Genetics', 'subject': 'Science', 'difficulty': 75},
    {'title': 'American Literature', 'subject': 'English', 'difficulty': 70},
    {'title': 'Renaissance History', 'subject': 'History', 'difficulty': 80},
    {'title': 'Multivariable Calculus', 'subject': 'Math', 'difficulty': 95},
    {'title': 'Quantum Mechanics', 'subject': 'Science', 'difficulty': 90},
    {'title': 'Technical Writing', 'subject': 'English', 'difficulty': 80},
    {'title': 'Colonial History', 'subject': 'History', 'difficulty': 65},
    {'title': 'Linear Algebra', 'subject': 'Math', 'difficulty': 85},
    {'title': 'Biochemistry', 'subject': 'Science', 'difficulty': 85},
    {'title': 'Drama Studies', 'subject': 'English', 'difficulty': 75},
    {'title': 'Contemporary History', 'subject': 'History', 'difficulty': 70},
    {'title': 'Probability Theory', 'subject': 'Math', 'difficulty': 80},
    {'title': 'Molecular Biology', 'subject': 'Science', 'difficulty': 80},
    {'title': 'Rhetoric and Composition', 'subject': 'English', 'difficulty': 70},
    {'title': 'Industrial Revolution', 'subject': 'History', 'difficulty': 75},
    {'title': 'Complex Analysis', 'subject': 'Math', 'difficulty': 90},
    {'title': 'Physical Chemistry', 'subject': 'Science', 'difficulty': 85},
    {'title': 'American Fiction', 'subject': 'English', 'difficulty': 75},
    {'title': 'History of the Americas', 'subject': 'History', 'difficulty': 70},

]

# Feedback des étudiants
feedbacks = [
    {'title': 'Advanced Algebra', 'perceived_difficulty': 70, 'feedback': "The course is very challenging but rewarding."},
    {'title': 'Basic Physics', 'perceived_difficulty': 65, 'feedback': "I find the course moderately difficult."},
    {'title': 'Shakespeare', 'perceived_difficulty': 85, 'feedback': "Understanding the old English is quite tough."},
    {'title': 'Geometry', 'perceived_difficulty': 60, 'feedback': "The explanations are clear, but the problems can be tricky."},
    {'title': 'Chemistry', 'perceived_difficulty': 75, 'feedback': "The lab sessions are very helpful, but the theory is difficult."},
    {'title': 'World History', 'perceived_difficulty': 55, 'feedback': "Interesting course, the content is engaging and well presented."},
    {'title': 'Modern Literature', 'perceived_difficulty': 70, 'feedback': "The analysis of the texts is complex, but the discussions help."},
    {'title': 'Trigonometry', 'perceived_difficulty': 80, 'feedback': "The concepts are abstract and require a lot of practice."},
    {'title': 'Biology', 'perceived_difficulty': 65, 'feedback': "A lot of memorization is needed, but the visual aids are great."},
    {'title': 'American History', 'perceived_difficulty': 60, 'feedback': "The lectures are detailed, but the reading materials are dense."},
    {'title': 'Calculus', 'perceived_difficulty': 90, 'feedback': "Extremely difficult, especially the integrals and differential equations."},
    {'title': 'Physics II', 'perceived_difficulty': 85, 'feedback': "The advanced topics are very challenging but fascinating."},
    {'title': 'Literary Analysis', 'perceived_difficulty': 80, 'feedback': "Critical thinking is crucial, and the essays are demanding."},
    {'title': 'European History', 'perceived_difficulty': 70, 'feedback': "The course is comprehensive, but the dates and events are hard to remember."},
    {'title': 'Advanced Chemistry', 'perceived_difficulty': 90, 'feedback': "The course requires a strong foundation in basic chemistry concepts."},
       {'title': 'Introduction to Algebra', 'perceived_difficulty': 50, 'feedback': "Good for beginners, concepts are well explained."},
    {'title': 'Fundamentals of Biology', 'perceived_difficulty': 60, 'feedback': "The course is interesting but requires a lot of memorization."},
    {'title': 'English Grammar Basics', 'perceived_difficulty': 45, 'feedback': "Easy to follow, great for non-native speakers."},
    {'title': 'Ancient History', 'perceived_difficulty': 55, 'feedback': "The course content is vast, but very engaging."},
    {'title': 'Intermediate Geometry', 'perceived_difficulty': 70, 'feedback': "Some of the problems are challenging, but practice helps."},
    {'title': 'Organic Chemistry', 'perceived_difficulty': 80, 'feedback': "The labs are difficult, but they enhance understanding."},
    {'title': 'Victorian Literature', 'perceived_difficulty': 75, 'feedback': "The language and context require a lot of analysis."},
    {'title': 'Statistics', 'perceived_difficulty': 85, 'feedback': "The mathematical concepts are tough, but the examples help."},
    {'title': 'Environmental Science', 'perceived_difficulty': 65, 'feedback': "The course covers a lot of ground, but is very informative."},
    {'title': 'Asian History', 'perceived_difficulty': 70, 'feedback': "The lectures are detailed, but remembering all the events is hard."},
    {'title': 'Differential Equations', 'perceived_difficulty': 90, 'feedback': "Very complex, requires a solid understanding of calculus."},
    {'title': 'Astronomy', 'perceived_difficulty': 70, 'feedback': "Fascinating but the mathematical part is challenging."},
    {'title': 'Creative Writing', 'perceived_difficulty': 60, 'feedback': "The assignments are fun, but require a lot of creativity."},
    {'title': 'Middle Ages History', 'perceived_difficulty': 75, 'feedback': "Very detailed, the amount of information is overwhelming."},
    {'title': 'Thermodynamics', 'perceived_difficulty': 85, 'feedback': "The concepts are tough, but the experiments help in understanding."},
    {'title': 'Modern Poetry', 'perceived_difficulty': 65, 'feedback': "Understanding the themes is challenging, but rewarding."},
    {'title': 'Discrete Mathematics', 'perceived_difficulty': 80, 'feedback': "The proofs are difficult, but the examples are helpful."},
    {'title': 'Genetics', 'perceived_difficulty': 75, 'feedback': "A lot of detailed information, but the practicals are great."},
    {'title': 'American Literature', 'perceived_difficulty': 70, 'feedback': "The readings are extensive, but the analysis is insightful."},
    {'title': 'Renaissance History', 'perceived_difficulty': 80, 'feedback': "Very comprehensive, but the volume of content is huge."},
    {'title': 'Multivariable Calculus', 'perceived_difficulty': 95, 'feedback': "Extremely difficult, requires a lot of prior knowledge."},
    {'title': 'Quantum Mechanics', 'perceived_difficulty': 90, 'feedback': "Very abstract, but the practical applications are fascinating."},
    {'title': 'Technical Writing', 'perceived_difficulty': 80, 'feedback': "The assignments are demanding, but very useful."},
    {'title': 'Colonial History', 'perceived_difficulty': 65, 'feedback': "Interesting perspectives, but the readings are dense."},
    {'title': 'Linear Algebra', 'perceived_difficulty': 85, 'feedback': "Challenging, especially the matrix operations."},
    {'title': 'Biochemistry', 'perceived_difficulty': 85, 'feedback': "Very detailed, but the labs help in understanding the concepts."},
    {'title': 'Drama Studies', 'perceived_difficulty': 75, 'feedback': "Understanding the plays is tough, but the discussions help."},
    {'title': 'Contemporary History', 'perceived_difficulty': 70, 'feedback': "The course is detailed, but the recent events are easier to remember."},
    {'title': 'Probability Theory', 'perceived_difficulty': 80, 'feedback': "The concepts are abstract, but the examples make them clearer."},
    {'title': 'Molecular Biology', 'perceived_difficulty': 80, 'feedback': "Requires a lot of memorization, but the practicals are very useful."},
    {'title': 'Rhetoric and Composition', 'perceived_difficulty': 70, 'feedback': "The writing assignments are challenging, but improve skills."},
    {'title': 'Industrial Revolution', 'perceived_difficulty': 75, 'feedback': "Interesting course, but the amount of information is large."},
    {'title': 'Complex Analysis', 'perceived_difficulty': 90, 'feedback': "Very challenging, requires a deep understanding of the basics."},
    {'title': 'Physical Chemistry', 'perceived_difficulty': 85, 'feedback': "The theoretical part is tough, but the practicals are helpful."},
    {'title': 'American Fiction', 'perceived_difficulty': 75, 'feedback': "The themes are complex, but the discussions are enlightening."},
    {'title': 'History of the Americas', 'perceived_difficulty': 70, 'feedback': "The course is comprehensive, but the volume of content is large."},
]


# Création de l'instance du système
learning_system = PersonalizedLearning(students, course_materials)

# Ajustement de la difficulté basée sur les feedbacks
learning_system.analyze_feedback(feedbacks)
learning_system.adjust_difficulty(feedbacks)

# Clustering des étudiants et attribution des matériaux
learning_system.cluster_students(method='neural')
materials_for_students = learning_system.assign_materials()

# Affichage des matériaux attribués par groupe
print("Matériaux attribués par groupe:", materials_for_students)

# Vérification du groupe de l'élève 1
student_cluster = learning_system.student_data[learning_system.student_data['StudentID'] == 1]['Cluster'].values[0]
print(f"L'élève 1 est attribué au groupe {student_cluster}")

# Evaluation d'un élève (par exemple, l'élève 1)
completed_materials = ['Advanced Algebra', 'Basic Physics', 'Shakespeare']
accuracy = learning_system.evaluate(student_id=1, completed_materials=completed_materials)

print(f"Précision de l'élève 1 avec les matériaux complétés: {accuracy:.2f}")

# Affichage des étudiants par groupe
learning_system.display_students_by_cluster()

# Recommandation des matériaux de cours pour un étudiant spécifique
student_id = 1
recommendations = learning_system.recommend_materials(student_id)
print(f"Recommandations pour l'élève {student_id}:", recommendations)
