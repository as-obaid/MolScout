import gov.nih.ncats.molvec.Molvec;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Keeps one JVM running for MolVec: reads one image path per line on stdin and writes one line
 * per image on stdout, "OK\t<path>\t<molfile>" or "ERR\t<path>\t<message>". The path is echoed so
 * the caller can check each reply belongs to its request. Fields escape backslash, newline,
 * carriage return and tab as \\, \n, \r and \t; a null molfile is sent as an empty one.
 */
public final class MolvecBatch {

    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true, "UTF-8");
        // MolVec prints debug lines to System.out; send them to stderr so stdout stays one line per image.
        System.setOut(System.err);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String path;
        while ((path = in.readLine()) != null) {
            String reply;
            try {
                String molfile = Molvec.ocr(new File(path));
                reply = "OK\t" + escape(path) + "\t" + escape(molfile == null ? "" : molfile);
            } catch (Throwable error) {
                reply = "ERR\t" + escape(path) + "\t" + escape(error.toString());
            }
            out.println(reply);
        }
    }

    private static String escape(String text) {
        StringBuilder escaped = new StringBuilder(text.length() + 64);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '\\': escaped.append("\\\\"); break;
                case '\n': escaped.append("\\n"); break;
                case '\r': escaped.append("\\r"); break;
                case '\t': escaped.append("\\t"); break;
                default: escaped.append(c);
            }
        }
        return escaped.toString();
    }
}
