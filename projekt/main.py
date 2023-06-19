import csv
import fileinput
import sqlite3
import tkinter as tk
import pickle
import numpy as np
import pandas as pd
from tkinter import messagebox, scrolledtext, simpledialog, filedialog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("App")

        self.data_graph_button = tk.Button(
            self.root, text="Data graph", width=25
        )
        self.data_graph_button.pack()

        self.choose_button = tk.Button(
            self.root, text="Build model", width=25, command=self.chooseRightModel
        )
        self.choose_button.pack()

        self.train_button = tk.Button(
            self.root, text="Train model", width=25, command=self.chooseRightModel
        )
        self.train_button.pack()

        self.test_button = tk.Button(
            self.root, text="Test model", width=25, command=self.testModel
        )
        self.test_button.pack()

        self.predict_button = tk.Button(
            self.root, text="Predict score", width=25, command=self.predictScore
        )
        self.predict_button.pack()

        self.add_data_button = tk.Button(
            self.root, text="Add data", width=25, command=self.addData
        )
        self.add_data_button.pack()

        self.save_to_database_button = tk.Button(
            self.root, text="Save database", width=25, command=self.saveToDatabase
        )
        self.save_to_database_button.pack()

        self.load_from_database_button = tk.Button(
            self.root, text="Load database", width=25, command=self.loadFromDatabase
        )
        self.load_from_database_button.pack()

        self.data_table = scrolledtext.ScrolledText(
            self.root, height=50, width=120
        )
        self.data_table.pack()

        scroll = tk.Scrollbar(self.root)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.configure(yscrollcommand=scroll.set)
        scroll.configure(command=self.data_table.yview)

        self.data = None
        self.dataHelp = None
        self.model = self.giveModel()
        self.loadData()

    def loadData(self):
        url = "wine.csv"
        self.data = pd.read_csv(url)
        self.dataHelp = self.data
        self.showData()

    def showData(self):
        self.data_table.delete("1.0", tk.END)
        df = pd.DataFrame(self.data)
        self.data_table.insert(tk.END, df.to_string(index=False))

    def chooseRightModel(self):
        self.data = pd.DataFrame(self.data)
        lista = self.data.iloc[:, 1:]
        wynik = self.data.iloc[:, 0]

        x_train, x_test, y_train, y_test = train_test_split(lista, wynik, test_size=0.25, random_state=2023)
        knn = KNeighborsClassifier(n_neighbors=5)

        knn.fit(x_train, y_train)
        param_grid = {'n_neighbors': list(range(1, 21)), 'metric': ['euclidean', 'manhattan']}

        grid_search = GridSearchCV(knn, param_grid, cv=KFold(n_splits=5, random_state=2023, shuffle=True))
        grid_search.fit(x_train, y_train)

        self.model = grid_search.best_estimator_

        self.saveModel()

    def testModel(self):
        self.data = pd.DataFrame(self.data)
        lista = self.data.iloc[:, 1:]
        wynik = self.data.iloc[:, 0]
        predictions = self.model.predict(lista)

        messagebox.showinfo("Informacja", "Ocena modelu : " + str(accuracy_score(wynik, predictions)))

    def predictScore(self):
        path = filedialog.askopenfilename()
        wynik = np.genfromtxt(path, delimiter=",")
        lista = wynik[:, 1:]
        prediction = self.model.predict(lista)
        for x in prediction:
            messagebox.showinfo("Informacja", "Przewidziano " + str(x))

    def saveModel(self):
        najlepszy_model = "najlepszy_model.pkl"
        with open(najlepszy_model, 'wb') as file:
            pickle.dump(self.model, file)

    def giveModel(self):
        najlepszy_model = "najlepszy_model.pkl"
        with open(najlepszy_model, 'rb') as file:
            return pickle.load(file)

    def addData(self):
        nazwa_pliku = "wine.csv"
        wynik = simpledialog.askstring("Wprowadz obiekt", "Info")
        with open(nazwa_pliku, "a") as file:
            file.write(wynik + "\n")

    def addHelpData(self):
        nazwa_pliku = "help.csv"
        wynik = simpledialog.askstring("Wprowadz obiekt", "Info")
        with open(nazwa_pliku, "w") as file:
            file.write(wynik + "\n")

    def saveToDatabase(self):
        mydb = sqlite3.connect('wines.db')
        my_cursor = mydb.cursor()

        my_cursor.execute(f"DROP TABLE wines")
        mydb.commit()

        my_cursor.execute(f"CREATE TABLE IF NOT EXISTS wines"
                          f"(label REAL, z1 REAL, z2 REAL, z3 REAL,z4 REAL,"
                          f"z5 REAL, z6 REAL, z7 REAL, z8 REAL, z9 REAL,"
                          f"z10 REAL, z11 REAL, z12 REAL, z13 REAL)")
        mydb.commit()

        with open('wine.csv', 'r') as file:
            csv_data = csv.reader(file)
            next(csv_data)

            for row in csv_data:
                my_cursor.execute('INSERT INTO wines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', row)

        mydb.commit()
        mydb.close()

    def loadFromDatabase(self):
        mydb = sqlite3.connect('wines.db')
        my_cursor = mydb.cursor()
        my_cursor.execute(f"SELECT * FROM wines")
        self.data = np.array(my_cursor.fetchall())

        my_cursor.close()
        mydb.close()

        self.showData()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
