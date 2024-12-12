package com.hadoop.mapreduce;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;

public class ConvertInput {
    public static class ConvertMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context con) throws IOException, InterruptedException {
            String[] content = value.toString().split("\\s++");
            String doc_id = content[1];
            String term_id = content[0];
            String tf_idf = content[2];
            con.write(new Text(doc_id), new Text(term_id + ":" + tf_idf));
        }
    } 

    public static class ConvertReducer extends Reducer<Text, Text, Text, Text>
    {
        @Override
        public void reduce(Text key, Iterable<Text> value, Context con) throws IOException, InterruptedException {
            Configuration conf = con.getConfiguration();
            Integer total_words = Integer.parseInt(conf.get("num_words"));
            String doc_tf_idf = "";
            int counter = 0;
            for (Text val:value) {
                counter += 1;
                doc_tf_idf += val.toString() + ",";
            }
            if (counter < total_words) {
                for (int i = 0; i < total_words - counter - 1; i++) {
                    doc_tf_idf += ":,";
                }
                doc_tf_idf += ":";
            }
            // FileSystem fs = FileSystem.get(conf);
            // Path test_path = new Path("/test/amogus.txt");
            // byte[] content = "christianronald".getBytes();
            // OutputStream os = fs.append(test_path);
            // os.write(content);
            // os.close();   
            con.write(key, new Text(doc_tf_idf));
        }
    }
}
