package com.hadoop.mapreduce;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.util.Random;
import org.apache.hadoop.conf.Configuration;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
public class ScalableKMeans {
    public static class ScalableKMeansMapper extends Mapper<Object, Text, Text, Text> {
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

        public void map(Object key, Text value, Context con) throws IOException, InterruptedException {
            String[] content = value.toString().split("\\s++"); 
            String[] doc_content = content[1].split(",");
            
            Double[] document = get_tf_idf(doc_content);
            String document_output = "";
            for (int i = 0; i < document.length; i++) {
                document_output += document[i].toString() + " ";
            }
            Random rd = new Random();
            Integer rand_key = rd.nextInt(5);
            // System.out.println(rand_key);
            con.write(new Text(rand_key.toString()), new Text(document_output));
        }
    }

    public static class ScalableKMeansReducer extends Reducer<Text, Text, Text, Text> {
        public Double vector_length(Double[] vector) {
            Double squared = (Double) 0.0;
            for (int i = 0; i < vector.length; i++) {
                squared += vector[i] * vector[i];
            }
            
            return (Double) Math.sqrt((double)squared);
        }

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

        public Double[] convert_string(String document_vector) {
            String[] values = document_vector.split(" ");
            Double[] new_vec = new Double[values.length];
            for (int i = 0; i < new_vec.length; i++) {
                new_vec[i] = Double.parseDouble(values[i]);
            }
            return new_vec;
        }

        public Integer[] scalable_initialization(ArrayList<ArrayList<Double>> data, Double l) {
            Random rd = new Random();
            int random_index = rd.nextInt(data.size());
            int num_elements = data.get(0).size();
            Double[] init_C = data.get(random_index).toArray(new Double[num_elements]);
            ArrayList<ArrayList<Double>> C = new ArrayList<>();
            C.add(data.remove(random_index));
            Double sum_dist = 0.0;
            for (int i = 0; i < data.size(); i++) {
                Double[] row = data.get(i).toArray(new Double[num_elements]);
                sum_dist += cosine_similarity(init_C, row);
            }
            int iteration = (int) Math.ceil(Math.log(sum_dist));
            for (int i = 0; i < iteration; i++) {
                Double[] psi_dist = new Double[num_elements];
                Arrays.fill(psi_dist, (double) 0.0);
                Double psi_sum_dist = (double)0.0;
                for (int i_1 = 0; i_1 < data.size(); i_1++) {
                    for (int i_2 = 0; i_2 < C.size(); i_2++) {
                        Double dist = cosine_similarity(C.get(i_2).toArray(new Double[num_elements]), data.get(i_1).toArray(new Double[num_elements]));
                        if (dist > psi_dist[i_1]) {
                            psi_dist[i_1] = dist;
                        }
                    }
                }

                for (int i_3 = 0; i_3 < psi_dist.length; i_3 ++) {
                    psi_sum_dist += psi_dist[i_3];
                }

                for (int j = 0; j < data.size(); j++) {
                    Double probability = (l * psi_dist[j]) / psi_sum_dist;
                    if (rd.nextDouble() < probability) {
                        C.add(data.remove(j));
                    }
                }
            }
            
            Integer[] w = new Integer[C.size()];
            Arrays.fill(w, Integer.valueOf(0));
            for (int i = 0; i < data.size(); i++) {
                Double[] row = data.get(i).toArray(new Double[num_elements]);
                Double max_dist = -1.0;
                Integer max_index = 0;
                for (int j = 0; j < C.size(); j++) {
                    Double[] centroid = C.get(j).toArray(new Double[num_elements]);
                    Double dist = cosine_similarity(row, centroid);
                    if (dist > max_dist) {
                        max_dist = dist;
                        max_index = j;
                    }
                }
                w[max_index] += 1;
                // System.out.println("Added " + w[max_index]);
            }
            return w;
        }

        public Integer[][] getKItems(Integer[] w, int k) {
            Integer[][] pairs = new Integer[w.length][2];
            for (int i = 0; i < w.length; i++) {
                pairs[i][0] = w[i];
                pairs[i][1] = i;
            }
            Arrays.sort(pairs, (a, b) -> Integer.compare(b[0], a[0]));

            Integer[][] result = new Integer[k][2];
            for (int i = 0; i < k; i++) {
                result[i][0] = pairs[i][0];
                result[i][1] = pairs[i][1];
            }
            return result;
        }

        public void reduce(Text key, Iterable<Text> value, Context con) throws IOException, InterruptedException{
            ArrayList<ArrayList<Double>> data = new ArrayList<>();
            ArrayList<ArrayList<Double>> data_clone = new ArrayList<>();
            Configuration conf = con.getConfiguration();
            Double l = Double.parseDouble(conf.get("l"));
            Integer k = Integer.parseInt(conf.get("k"));
            for (Text v:value) {
                String document_vector = v.toString();
                Integer elements = document_vector.split(" ").length;
                Double[] document = convert_string(document_vector);
                ArrayList<Double> row = new ArrayList<>();
                for (int i = 0; i < elements; i++) {
                    row.add(document[i]);
                }
                data.add(row);
                data_clone.add(row);
            }
            int num_elements = data.get(0).size();
            Integer[] weights = scalable_initialization(data_clone, l);
            String output = "";
            Integer[][] sorted_weights = getKItems(weights, k);
            for (int i = 0; i < k; i++) {
                ArrayList<Double> centroid = data.get(sorted_weights[i][1]);
                for (int j = 0; j < num_elements; j++) {
                    output += centroid.get(j).toString();
                    if (j != num_elements - 1) {
                        output += " ";
                    }
                }
                if (i != k - 1) {
                    output += ",";
                }
            }
            con.write(key, new Text(output));
        }
    }
}
