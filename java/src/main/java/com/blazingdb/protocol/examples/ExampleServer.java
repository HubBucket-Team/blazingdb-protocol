package com.blazingdb.protocol.examples;

import com.blazingdb.protocol.BlazingConfiguration;
import com.blazingdb.protocol.Server;
import com.blazingdb.protocol.ServerHandler;
import com.google.flatbuffers.FlatBufferBuilder;

import java.io.File;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.List;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.List;
import blazingdb.protocol.*;

import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;
import java.nio.channels.Channels;
import org.apache.commons.io.IOUtils;

public class ExampleServer {
	public final static int BUFFER_SIZE = 4096 * 1024 * 4;
	public final static String PERMISSION_LABEL = "┌∩┐(x_x)┌∩┐";
	public final static String OFFLINE_MESSAGE = "The BlazingDB server is restarting please try again in a moment.";

	public final static String PERMISSIONS_DELIM = "┌∩┐(◣_◢)┌∩┐";
 

	public static void main(String[] args) throws IOException {

        java.io.File path = new java.io.File("/tmp/socket");
        path.deleteOnExit();

		Server server = new Server(path, new ServerHandler() {
			
			@Override
			public void accept(ByteBuffer buffer) {
				// TODO Auto-generated method stub
				BlazingQueryMessage serverMessage = new BlazingQueryMessage(buffer);

				System.out.println("## SQL Statement: " + serverMessage.statement() + "  |  with grank/revoke => "
						+ serverMessage.authorization());
				System.out.println("## The FlatBuffer was successfully created and verified!");

			}
		});

	}
}
