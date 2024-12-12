import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split("\\s+");
            if (tokens.length == 3) {
                String termId = tokens[0];
                int freq = Integer.parseInt(tokens[2]);
                context.write(new Text(termId), new IntWritable(freq));
            }
        }
    }

    public static class TopWordsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private Map<String, Integer> termFrequencyMap = new HashMap<>();
        private static final int TOP_N = 10;

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int totalFrequency = 0;
            for (IntWritable value : values) {
                totalFrequency += value.get();
            }
            termFrequencyMap.put(key.toString(), totalFrequency);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Sắp xếp và ghi kết quả
            List<Map.Entry<String, Integer>> sortedEntries = new ArrayList<>(termFrequencyMap.entrySet());
            sortedEntries.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));

            int count = 0;
            for (Map.Entry<String, Integer> entry : sortedEntries) {
                if (count < TOP_N) {
                    context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                    count++;
                } else {
                    break;
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Top Words");

        FileSystem fs = FileSystem.get(conf);
        Path outputPath = new Path(args[1]);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }

        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(TopWordsReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, outputPath);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

