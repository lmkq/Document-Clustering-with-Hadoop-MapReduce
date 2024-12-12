package com.hadoop.mapreduce;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.util.Arrays;
// import java.util.Random;
// import java.text.DecimalFormat;

public class KMeans {
    // Read new centroids from file every iteration
    // Format "centroid_id  tf_idf_1,tf_idf_2,..."     
    // Returns a serialized version of the centroid coordinates
    public static String read_centroids(Configuration conf, String file_path) throws IOException, InterruptedException{
        String centroid_string = "";

        FileSystem fs = FileSystem.get(conf);
        Path read_path = new Path(file_path);
        if (fs.exists(read_path)) {
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(read_path)));
            String line;
            line = br.readLine();
            while (line != null) {
                centroid_string += line.split("\t")[1] + ",";
                line = br.readLine();
            }
            br.close();
        }
        return centroid_string;
    }

    public static Double[][] scalable_kmeans_init(Configuration conf, String file_path, Integer k, Integer elements) throws IOException, InterruptedException {
        Double[][] tmp = new Double[k][elements];
        for (Double[] row:tmp) {
            Arrays.fill(row, Double.valueOf(0));
        }
        Integer num_workers = Integer.parseInt(conf.get("num_workers"));
        FileSystem fs = FileSystem.get(conf);
        Path read_path = new Path(file_path);
        
        if (fs.exists(read_path)) {
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(read_path)));
            String line;
            line = br.readLine();
            while (line != null) {
                if (line.length() == 0) {
                    break;
                }
                String[] centroids = line.split("\t")[1].split(",");
                for (int i = 0; i < k; i++) {
                    String[] tf_idf = centroids[i].split(" ");
                    for (int j = 0; j < elements; j++) {
                        tmp[i][j] += Double.parseDouble(tf_idf[j]);
                    }
                }
                line = br.readLine();
            }
            br.close();
        }
        String tmp_output = "";
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < elements; j++) {
                tmp[i][j] /= num_workers;
                tmp_output += tmp[i][j].toString() + " ";
            }
            tmp_output += "\n";
        }
        FileWriter writer = new FileWriter("test.txt");
        writer.write(tmp_output);
        writer.close();
        return tmp;
    }

    public static void main(String args[]) throws IOException, ClassNotFoundException, InterruptedException {
        // Configure important variables here
        String tmp_output_path = args[1] + "/tmp/";
        String output_path = args[1] + "/final/";
        final Integer num_clusters = 5;
        final Integer iterations = 10;
        final Double l = (double)num_clusters * 10;
        
        // Count number of unique words for array initialization
        Configuration conf_1 = new Configuration();
        Job unique_words = Job.getInstance(conf_1, "Count unique words");
        unique_words.setJarByClass(UniqueWords.class);
        unique_words.setMapperClass(UniqueWords.UniqueWordsMapper.class);
        unique_words.setReducerClass(UniqueWords.UniqueWordsReducer.class);
        unique_words.setMapOutputKeyClass(Text.class);
        unique_words.setMapOutputValueClass(IntWritable.class);
        unique_words.setOutputKeyClass(Text.class);
        unique_words.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(unique_words, new Path(args[0]));
        FileOutputFormat.setOutputPath(unique_words, new Path(tmp_output_path + "/uniq"));
        unique_words.waitForCompletion(true);
        Integer num_words = (int) unique_words.getCounters().findCounter(WordCounter.wordcounter.counter).getValue();
        
        //  Refactor input to suggested form, where every input line is "doc_id  term_id:tf_idf,..."
        Configuration conf_2 = new Configuration();
        System.out.println(num_words);
        conf_2.set("num_words", num_words.toString());
        Job convert_input = Job.getInstance(conf_2, "Refactor input");
        convert_input.setJarByClass(ConvertInput.class);
        convert_input.setMapperClass(ConvertInput.ConvertMapper.class);
        convert_input.setReducerClass(ConvertInput.ConvertReducer.class);
        convert_input.setMapOutputKeyClass(Text.class);
        convert_input.setMapOutputValueClass(Text.class);
        convert_input.setOutputKeyClass(Text.class);
        convert_input.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(convert_input, new Path(args[0]));
        // K-Means job input should be read from here
        FileOutputFormat.setOutputPath(convert_input, new Path(tmp_output_path + "/refactor"));
        convert_input.waitForCompletion(true);

        Configuration conf_3 = new Configuration();
        System.out.println(num_words);
        conf_3.set("k", num_clusters.toString());
        conf_3.set("l", l.toString());
        conf_3.set("num_workers", "5");
        Job centroid_init = Job.getInstance(conf_3, "Task 2_3 init centroids");
        centroid_init.setJarByClass(ScalableKMeans.class);
        centroid_init.setMapperClass(ScalableKMeans.ScalableKMeansMapper.class);
        centroid_init.setReducerClass(ScalableKMeans.ScalableKMeansReducer.class);
        centroid_init.setMapOutputKeyClass(Text.class);
        centroid_init.setMapOutputValueClass(Text.class);
        centroid_init.setOutputKeyClass(Text.class);
        centroid_init.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(centroid_init, new Path(tmp_output_path + "/refactor/part-r-00000"));
        // K-Means job input should be read from here
        FileOutputFormat.setOutputPath(centroid_init, new Path(tmp_output_path + "/init_centroids"));
        centroid_init.waitForCompletion(true);
        
        Double[][] centroids = new Double[num_clusters][num_words];
        // DecimalFormat df = new DecimalFormat();
        String centroid_string = "";
        // Random rd = new Random();
        // 2.2 centroid initialization
        // for (int j = 0; j < num_clusters; j++) {
        //     for (int t = 0; t < num_words; t++)
        //     {
        //         centroids[j][t] = Double.valueOf(df.format(Math.abs(rd.nextDouble())));
        //         centroid_string += centroids[j][t].toString() + " ";
        //     }
        //     centroid_string += ",";
        // }
        // 2.3 centroid_initialization
        centroids = scalable_kmeans_init(conf_3, tmp_output_path + "/init_centroids/part-r-00000", num_clusters, num_words);
        for (int j = 0; j < num_clusters; j++) {
            for (int t = 0; t < num_words; t++)
            {
                centroid_string += centroids[j][t].toString() + " ";
            }
            centroid_string += ",";
        }
        // System.out.println(centroid_string);
        // System.out.println("Number of centroids: " + centroid_string.split(",").length);
        for (Integer i = 0; i < iterations; i++) {
            String centroid_update = output_path + "task_2_3_iter" + i + ".clusters";
            Configuration conf_main = new Configuration();
            conf_main.set("num_clusters", num_clusters.toString());
            conf_main.set("centroids", centroid_string);
            conf_main.set("output_folder", output_path);
            conf_main.set("iteration", i.toString());
            conf_main.set("max_iteration", iterations.toString());
            conf_main.set("has_written_mean", "0");
            conf_main.set("has_written_loss", "0");
            Job kmeans = Job.getInstance(conf_main, "Iteration: " + i);
            kmeans.setJarByClass(KMeans.class);
            kmeans.setMapperClass(KMeansMapper.class);
            kmeans.setReducerClass(KMeansReducer.class);
            kmeans.setMapOutputKeyClass(Text.class);
            kmeans.setMapOutputValueClass(Text.class);
            kmeans.setOutputKeyClass(Text.class);
            kmeans.setOutputValueClass(Text.class);
            FileInputFormat.addInputPath(kmeans, new Path(tmp_output_path + "/refactor/part-r-00000"));
            // Dummy save file
            FileOutputFormat.setOutputPath(kmeans, new Path(output_path + "/kmeans_iter_" + i));
            boolean success = kmeans.waitForCompletion(true);
            if (!success) {
                System.out.println("Failure at iteration " + i);
                System.exit(1);
            }
            // Read new centroids
            centroid_string = read_centroids(conf_main, centroid_update);
        }


    }
}