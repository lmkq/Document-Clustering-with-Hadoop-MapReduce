package com.hadoop.mapreduce;


import java.io.IOException;
import java.util.Arrays;


import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

public class KMeansMapper extends Mapper<Object, Text, Text, Text>{
    // Returns the 2D array representation of the centroids
    public Double[][] decode_centroids(String centroid_string) {
        String[] initial_centroids = centroid_string.split(",");
        Integer k = initial_centroids.length;
        String[] sample_centroid = initial_centroids[0].split(" ");
        Double[][] centroids = new Double[k][sample_centroid.length];
        for (int i = 0; i < k; i++) {
            String[] ith_centroid = initial_centroids[i].split(" ");
            for (int j = 0; j < sample_centroid.length; j++) {
                centroids[i][j] = Double.parseDouble(ith_centroid[j]);
            }
        }
        return centroids;
    }

    // Utility function for cosine similarity
    public Double vector_length(Double[] vector) {
        Double squared = (Double) 0.0;
        for (int i = 0; i < vector.length; i++) {
            squared += vector[i] * vector[i];
        }
        
        return (Double) Math.sqrt((double)squared);
    }

    // Calculate cosine similarity between centroid and sample
    public Double cosine_similarity(Double[] vector_a, Double[] vector_b) {
        if (vector_a.length != vector_b.length) {
            return - (Double) 9999.0;
        }
        if (vector_a.equals(vector_b)) {
            return (Double) 1.0;
        }

        Double dot_product = (Double) 0.0;
        Double sum_vector_length = (Double) 0.0;
        for (int i = 0; i < vector_a.length; i++) {
            dot_product += vector_a[i] * vector_b[i];
        }
        sum_vector_length = vector_length(vector_a) + vector_length(vector_b);
        if (sum_vector_length == 0.0)
            return - (Double) 9999.0;
        return (Double) dot_product / sum_vector_length;
    }


    // Get tf_idf score from the refactored input
    public Double[] get_tf_idf(String[] list_inputs) {
        Double[] result = new Double[list_inputs.length];

        int eol = 0;
        for (int i = 0; i < result.length; i++) {
            String[] pair = list_inputs[i].split(":");
            if (pair.length == 0) {
                eol = i;
                break;
            }
            result[i] = Double.parseDouble(pair[1]);
        }
        Arrays.fill(result, eol, result.length, (double)0);
        return result;
    }

    // Assign each document to a centroid, return the centroid id
    public Integer assign_cluster(Double[] document, Double[][] centroids) {
        Double max_dist = - (double) 1;
        Integer label = 0;
        for (int i = 0; i < centroids.length; i++) {
            Double dist = cosine_similarity(centroids[i], document);
            if (dist > max_dist) {
                    max_dist = dist;
                    label = i;
            }
        }
        return label;
    }


    public void map(Object key, Text value, Context con) throws IOException, InterruptedException {
        String[] content = value.toString().split("\\s++"); 
        String doc_id = content[0];
        String[] doc_content = content[1].split(",");
        Configuration conf = con.getConfiguration();
        Double[][] centroids = decode_centroids(conf.get("centroids"));
        // System.out.println(con.getConfiguration().get("iteration") + " " +centroids.length + " " + centroids[0].length);
        Double[] document = get_tf_idf(doc_content);
        Integer cluster_center = assign_cluster(document, centroids);
        String document_output = doc_id + ",";
        for (int i = 0; i < document.length; i++) {
            document_output += document[i].toString() + " "; 
        }
        con.write(new Text(cluster_center.toString()), new Text(document_output));
    }
}
