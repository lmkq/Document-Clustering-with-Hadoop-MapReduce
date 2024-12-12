package com.hadoop.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.Text;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NavigableMap;
import java.util.TreeMap;

public class KMeansReducer extends Reducer<Text, Text, Text, Text> {
    // Convert serialized version of coordinates to an array of double
    public Double[] convert_string(String document_vector) {
        String[] values = document_vector.split(" ");
        Double[] new_vec = new Double[values.length];
        for (int i = 0; i < new_vec.length; i++) {
            new_vec[i] = Double.parseDouble(values[i]);
        }
        return new_vec;
    }

    // Write the centroids to an output file, deleting the old iterations every new iteration
    public void write_centroids(Configuration conf, String cluster_id, String centroids, int iteration) throws IOException, InterruptedException {
        FileSystem fs = FileSystem.get(conf);
        Path old_path = new Path(conf.get("output_folder") + "task_2_2_iter" + (iteration - 1) + ".clusters");
        if (fs.exists(old_path)) {
            fs.delete(old_path, true);          
        }
        Path write_path;
        if (iteration == (Integer.parseInt(conf.get("max_iteration"))) - 1) {
            write_path = new Path(conf.get("output_folder") + "task_2_2.clusters");
        }   
        else {
            write_path = new Path(conf.get("output_folder") + "task_2_2_iter" + iteration + ".clusters");
        }
        String content = cluster_id + "\t" + centroids + "\n";
        byte[] byte_content = content.getBytes();
        OutputStream os;
        if (fs.exists(write_path)) {
            os = fs.append(write_path);
        }
        else {
            os = fs.create(write_path);
        }
        os.write(byte_content);
        os.close();
    }

    // Write the document id and their assigned cluster id to the file, deleting the old iterations every new iteration
    public void write_document_clustering(Configuration conf, String cluster_id, String doc_id, int iteration) throws IOException, InterruptedException {
        FileSystem fs = FileSystem.get(conf);
        Path old_path = new Path(conf.get("output_folder") + "task_2_2_iter" + (iteration - 1) + ".classes");
        if (fs.exists(old_path)) {
            fs.delete(old_path, true);          
        }
        Path write_path;
        if (iteration == (Integer.parseInt(conf.get("max_iteration")) - 1)) {
            write_path = new Path(conf.get("output_folder") + "task_2_2.classes");
        }   
        else {
            write_path = new Path(conf.get("output_folder") + "task_2_2_iter" + iteration + ".classes");
        }
        String content = "";
        String[] list_docs = doc_id.split(" ");
        for (int i = 0; i < list_docs.length; i++) {
            content += list_docs[i] + " " + cluster_id + "\n";
        }
        byte[] byte_content = content.getBytes();
        OutputStream os;
        if (fs.exists(write_path)) {
            os = fs.append(write_path);
        }
        else {
            os = fs.create(write_path);
        }
        os.write(byte_content);
        os.close();
    }


    // Write the top 10 tf_idf values from the centroid
    public void write_top_10(Configuration conf, String cluster_id, String top_10, int iteration) throws IOException, InterruptedException{
        FileSystem fs = FileSystem.get(conf);
        String content = "";
        if (conf.get("has_written_mean").toString() == "0") {
            content += "\nIteration " + (iteration + 1) + ": \n";
            conf.set("has_written_mean", "1");
        }
        Path write_path = new Path(conf.get("output_folder") + "task_2_2.txt");
        content += top_10 + "\n";
        byte[] byte_content = content.getBytes();
        OutputStream os;
        if (fs.exists(write_path)) {
            os = fs.append(write_path);
        }
        else {
            os = fs.create(write_path);
        }
        os.write(byte_content);
        os.close();
    }

    // Write WCSS loss to the file
    public void write_losses(Configuration conf, String cluster_id, Double loss, int iteration) throws IOException, InterruptedException {
        FileSystem fs = FileSystem.get(conf);
        String content = "";
        if (conf.get("has_written_loss").toString() == "0") {
            content += "\nIteration " + (iteration + 1) + ": \n";
            conf.set("has_written_loss", "1");
        }
        Path write_path = new Path(conf.get("output_folder") + "task_2_2.losses");
        content += loss.toString();
        content += "\n";
        byte[] byte_content = content.getBytes();
        OutputStream os;
        if (fs.exists(write_path)) {
            os = fs.append(write_path);
        }
        else {
            os = fs.create(write_path);
        }
        os.write(byte_content);
        os.close();
    }

    public Double calculate_loss(Double[] data, Double[] centroid) {
        Double wcss = (double) 0;
        for (int i = 0; i < data.length; i++) {
            wcss += Math.pow(centroid[i] - data[i], 2);
        }
        return wcss;
    }
    
    public String get_top_10(Double[] centroid) {
        TreeMap<Double, Integer> top_10 = new TreeMap<Double, Integer>();
        for (int i = 0; i < centroid.length; i++) {
            top_10.put(centroid[i], i);
        }
        String top_10_mean = "";
        NavigableMap<Double, Integer> descending_top_10 = top_10.descendingMap();
        Iterator tmp = descending_top_10.entrySet().iterator();
        for (int i = 0; i < 10; i++) {
            top_10_mean += tmp.next().toString() + ", ";
        }
        return top_10_mean;
    }


    public void reduce(Text key, Iterable<Text> value, Context con) throws IOException, InterruptedException {
        Configuration conf = con.getConfiguration();
        Integer iteration = Integer.parseInt(conf.get("iteration"));
        String old_centroids = conf.get("centroids");
        String[] num_centroids = old_centroids.split(",");
        int elements = num_centroids[0].split(" ").length;
        Double[] new_centroid = new Double[elements];
        Arrays.fill(new_centroid, Double.valueOf(0));
        int counter = 0;
        String list_docs = "";
        String cluster_id = key.toString();
        Double[] data = new Double[elements];
        Double wcss = (double) 0;
        for (Text v: value) {
            String[] true_val = v.toString().split(",");
            String doc_id = true_val[0];
            list_docs += doc_id + " ";
            String doc_content = true_val[1];
            Double[] tmp = convert_string(doc_content);
            for (int i = 0; i < tmp.length; i++) {
                data[i] = tmp[i];
            }
            counter += 1;
            for (int i = 0; i < new_centroid.length; i++) {
                new_centroid[i] += tmp[i];
            }
            wcss += calculate_loss(data, new_centroid);
        }
        String new_centroid_str = "";
        for (int i = 0; i < new_centroid.length; i++) {
            new_centroid[i] /= (counter + 1);
            new_centroid_str += new_centroid[i].toString() + " ";
        }
        String top_10 = get_top_10(new_centroid);
        write_centroids(conf, cluster_id, new_centroid_str, iteration);
        write_document_clustering(conf, cluster_id, list_docs, iteration);
        write_losses(conf, cluster_id, wcss, iteration);
        write_top_10(conf, cluster_id, top_10, iteration);
        con.write(key, new Text(new_centroid_str));
    }
}