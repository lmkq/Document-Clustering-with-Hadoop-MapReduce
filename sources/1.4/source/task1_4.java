import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.io.*;

import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FSDataOutputStream;

public class task1_4 {

    // save the number of total document to config
    public static void saveToTalDocumenttoConfig(String filePath, Configuration conf) {
        Path path = new Path(filePath);

        try {
            FileSystem fs = FileSystem.get(conf);
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
                String line;
                int lineCount = 0;

                while ((line = br.readLine()) != null) {
                    lineCount++;
                    if (lineCount == 2) { // 2nd line in input file contains the matrix size
                        String[] parts = line.trim().split("\\s+");
                        int totalDocs = Integer.parseInt(parts[1]);
                        conf.setInt("totalDocs", totalDocs);
                        break;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // compute Term Frequency
    public static class TermFrequencyMapper extends Mapper<LongWritable, Text, Text, Text> {
        private int row = 0;

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // skip the first two lines
            if (row < 2) {
                row++;
            } else {
                String[] parts = value.toString().split("\\s+");
                // key: docID, value: termID+frequency
                context.write(new Text(parts[1]), new Text(parts[0] + "\t" + parts[2]));
            }
        }
    }

    public static class TermFrequencyReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            double total = 0;
            Map<String, Integer> termFrequencyMap = new HashMap<>();

            for (Text value : values) { // compute total words per document
                String[] parts = value.toString().split("\\s+");
                int frequency = Integer.parseInt(parts[1]);
                total += frequency;
                termFrequencyMap.put(parts[0], frequency);
            }

            // compute TF
            for (Map.Entry<String, Integer> entry : termFrequencyMap.entrySet()) {
                String termID = entry.getKey();
                int termFrequency = entry.getValue();
                double tf = termFrequency / total;
                context.write(new Text(key.toString() + "\t" + termID), new Text(String.format("%.6f", tf)));
            }
        }
    }

    public static class TFIDFMapper extends Mapper<LongWritable, Text, Text, Text> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");

            // key: termID, value: docID+TF
            context.write(new Text(parts[0]), new Text(parts[1] + "\t" + parts[2]));
        }
    }

    public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {
        private int totalDocs;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            totalDocs = context.getConfiguration().getInt("totalDocs", 1); // get totalDocs
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double documentCount = 0;
            List<String> docTFList = new ArrayList<>();

            for (Text value : values) { // get the document count: number of document contains term
                documentCount += 1;
                docTFList.add(value.toString());
            }

            double idf = Math.log(totalDocs / documentCount); // compute IDF

            for (String docTF : docTFList) {
                String[] parts = docTF.split("\\s+");

                String docId = parts[0];
                double tf = Double.parseDouble(parts[1]);

                double tfidf = tf * idf; // compute TFIDF

                context.write(new Text(key.toString() + "\t" + docId), new Text(String.format("%.6f", tfidf)));
            }
        }
    }

    // write the final mtx file
    public static void writeMtxFile(Configuration conf, String hdfsOutputPath, String localOutputPath)
            throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path outputPath = new Path(hdfsOutputPath);

        // get list of files in the output directory
        FileStatus[] fileStatuses = fs.listStatus(outputPath);
        LinkedHashSet<String> entries = new LinkedHashSet<>();

        // read files from HDFS
        for (FileStatus status : fileStatuses) {
            Path filePath = status.getPath();
            FSDataInputStream inputStream = fs.open(filePath);
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            String line;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\s+");
                int termId = Integer.parseInt(parts[0]);
                int docId = Integer.parseInt(parts[1]);
                double tf = Double.parseDouble(parts[2]);

                entries.add(String.format("%d %d %.6f", termId, docId, tf));
            }

            br.close();
            inputStream.close();
        }

        // write MTX file
        try (FileOutputStream fos = new FileOutputStream(localOutputPath)) {
            for (String entry : entries) {
                fos.write((entry + "\n").getBytes());
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        String inputPath = args[0];
        String outputPath = args[1];

        // save the number of total documents into config
        saveToTalDocumenttoConfig(inputPath + "/output_1_2.mtx", conf);

        // Job 1: Calculate TF
        Job tfJob = Job.getInstance(conf, "TF");
        tfJob.setJarByClass(task1_4.class);
        tfJob.setMapperClass(TermFrequencyMapper.class);
        tfJob.setReducerClass(TermFrequencyReducer.class);
        tfJob.setMapOutputKeyClass(Text.class);
        tfJob.setMapOutputValueClass(Text.class);

        FileInputFormat.addInputPath(tfJob, new Path(inputPath));

        FileOutputFormat.setOutputPath(tfJob, new Path("/tfJob"));

        if (!tfJob.waitForCompletion(true)) {
            System.exit(1);
        }

        // Job 2: Calculate TFIDF
        Job tfidfJob = Job.getInstance(conf, "TFIDF");

        tfidfJob.setJarByClass(task1_4.class);
        tfidfJob.setMapperClass(TFIDFMapper.class);
        tfidfJob.setReducerClass(TFIDFReducer.class);
        tfidfJob.setOutputKeyClass(Text.class);
        tfidfJob.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(tfidfJob, new Path("/tfJob"));
        FileOutputFormat.setOutputPath(tfidfJob, new Path(outputPath));

        if (!tfidfJob.waitForCompletion(true)) {
            System.exit(1);
        }

        writeMtxFile(conf, outputPath, "/home/lmkquynh/bd_lab2/task_1_4.mtx");
        System.exit(0);
    }
}