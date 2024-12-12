import java.util.*;
import java.text.DecimalFormat;
import java.io.IOException;
import javax.naming.Context;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.CreateFlag;

public class AverageTFIDF {
    public static class MapJob1 extends Mapper<Object, Text, Text, Text> {
        private List<String> docIDs = new ArrayList<String>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            Path docFilePath = new Path("/user/hadoop/input/bbc.docs");
            // Đọc bbc.docs
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(docFilePath)))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String trimmedWord = line.trim();
                    docIDs.add(trimmedWord);
                }
            }
        }

        private String getDoc(Integer ID) {
            return docIDs.get(ID - 1);
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split("\\s+");
            String doc = getDoc(Integer.valueOf(tokens[1]));
            String type = doc.substring(0, doc.length() - 4);
            context.write(new Text(type),
                    new Text(tokens[0] + "\t" + doc.substring(doc.length() - 4) + "\t" + tokens[2]));
        }
    }

    public static class ReduceJob1 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            HashSet<String> doc = new HashSet<>();
            HashMap<String, Double> data = new HashMap<>();

            for (Text val : values) {
                String[] tokens = val.toString().split("\\s+");
                doc.add(tokens[1]);
                if (!data.containsKey(tokens[0]))
                    data.put(tokens[0], Double.valueOf(tokens[2]));
                else
                    data.put(tokens[0], data.get(tokens[0]) + Double.valueOf(tokens[2]));
            }

            double count = doc.size();
            data.forEach((keyMap, value) -> {
                double result = (double) value / count;
                try {
                    context.write(new Text(keyMap + "\t" + key.toString()), new Text(Double.toString(result)));
                } catch (InterruptedException | IOException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    public static class MapJob2 extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split("\\s+");
            context.write(new Text(tokens[1]), new Text(tokens[0] + "\t" + tokens[2]));
        }
    }

    public static class ReduceJob2 extends Reducer<Text, Text, Text, Text> {
        private List<String> termIDs = new ArrayList<String>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            Path termFilePath = new Path("/user/hadoop/input/bbc.terms");
            // Đọc bbc.terms
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(termFilePath)))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String trimmedWord = line.trim();
                    termIDs.add(trimmedWord);
                }
            }
        }

        private String getTerm(Integer ID) {
            return termIDs.get(ID - 1);
        }

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            List<Pair<String, Double>> data = new ArrayList<>();

            for (Text val : values) {
                String[] tokens = val.toString().split("\\s+");
                String term = getTerm(Integer.valueOf(tokens[0]));
                data.add(new Pair<>(term, Double.valueOf(tokens[1])));
            }

            data.sort(new Comparator<Pair<String, Double>>() {
                @Override
                public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
                    return o2.getValue().compareTo(o1.getValue());
                }
            });

            int count = 0;
            DecimalFormat df = new DecimalFormat("#.##");
            String result = "";
            for (Pair<String, Double> pair : data) {
                if (count >= 5)
                    break;
                if (count != 0)
                    result += ", ";
                count++;
                String roundedString = df.format(pair.getValue());
                result += pair.getKey() + ":" + roundedString;
            }
            String output = key.toString().substring(0, 1).toUpperCase() + key.toString().substring(1) + ": " + result;
            context.write(new Text(output), new Text(" "));
        }

        static class Pair<K, V> {
            private K key;
            private V value;

            public Pair(K key, V value) {
                this.key = key;
                this.value = value;
            }

            public K getKey() {
                return key;
            }

            public V getValue() {
                return value;
            }
        }
    }

    public static void writeReducerOutput(Path result, Path out) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] fStatuses = fs.listStatus(result);
        FileContext fileContext = FileContext.getFileContext();
        FSDataOutputStream outStream = fileContext.create(out, EnumSet.of(CreateFlag.CREATE, CreateFlag.OVERWRITE));

        for (FileStatus status : fStatuses) {
            String line = new String();
            FSDataInputStream reduceOutputStream = fileContext.open(status.getPath());
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(reduceOutputStream));
            while ((line = bufferedReader.readLine()) != null) {
                outStream.writeBytes(line + "\n");
            }
            bufferedReader.close();
        }
        outStream.close();
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // Job 1
        Job job1 = Job.getInstance(conf, "Job1");
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path("/tempDir/1.5"));

        job1.setJarByClass(AverageTFIDF.class);
        job1.setMapperClass(MapJob1.class);
        job1.setReducerClass(ReduceJob1.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        // Đợi job 1 xong
        boolean success = job1.waitForCompletion(true);
        if (!success) {
            System.exit(1);
        }

        // Job 2
        Job job2 = Job.getInstance(conf, "top");
        FileInputFormat.addInputPath(job2, new Path("/tempDir/1.5"));
        FileOutputFormat.setOutputPath(job2, new Path("/user/hadoop/task_1_5.txt"));

        job2.setJarByClass(AverageTFIDF.class);
        job2.setMapperClass(MapJob2.class);
        job2.setReducerClass(ReduceJob2.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        success = job2.waitForCompletion(true);
        if (!success) {
            System.exit(1);
        }
        fs.delete(new Path("/tempDir/1.5"), true);
        System.exit(0);
    }
}
