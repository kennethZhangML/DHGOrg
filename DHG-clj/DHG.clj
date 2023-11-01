#!/usr/bin/env bb

(defn read-api-key [file-path]
  (slurp file-path))

(defn fetch-data [url headers]
  (:body (clojure.java.io/as-url url)
         {:headers headers :as :json}))

(defn create-nodes [issues commits prs]
  (concat
   (map #(hash-map :id (str "issue" (:id %)) :type "issue" :data %) issues)
   (map #(hash-map :id (str "commit_" (:sha %)) :type "commit" :data %) commits)
   (map #(hash-map :id (str "pushRequest_" (get-in % [:head :sha])) :type "pushRequest" :data %) prs)))

(defn create-hyperedges [issues commits]
  (concat
   (map #(hash-map :source [(str "user_" (get-in % [:user :id]))]
                   :target [(str "issue_" (:id %))]
                   :type "created_by"
                   :data {}) issues)
   (map #(hash-map :source [(str "user_" (get-in % [:committer :id]))]
                   :target [(str "commit_" (:sha %))]
                   :type "committed_by"
                   :data {}) commits)))

(defn construct-dhg [base-url headers dhg-name]
  (let [issues   (fetch-data (str base-url "issues") headers)
        commits  (fetch-data (str base-url "commits") headers)
        prs      (fetch-data (str base-url "pulls") headers)
        nodes    (create-nodes issues commits prs)
        hyperedges (create-hyperedges issues commits)]
    {:nodes nodes :hyperedges hyperedges}))

(defn save-dhg [dhg dhg-name]
  (spit (str "dihypergraph_" dhg-name ".json") (with-out-str (prn dhg))))

(defn main []
  (let [base-url "https://api.github.com/repos/kennethZhangML/StochFlow/"
        token (read-api-key "/Users/kennethzhang/Desktop/gh-apikey.txt")
        headers {"Authorization" (str "token " token)
                 "Accept" "application/vnd.github.v3+json"}
        dhg-name "stochflow"
        dhg (construct-dhg base-url headers dhg-name)]
    (save-dhg dhg dhg-name)
    (println (str "Dihypergraph constructed to dihypergraph_" dhg-name))))

(main)
