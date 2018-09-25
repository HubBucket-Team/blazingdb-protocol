package com.blazingdb.protocol;

import java.io.File;
import java.io.IOException;

public class BlazingConfiguration {

	public static String  unixSocketPath = "/tmp/socket";

	public static File unixSocket() throws IOException {
		final File unixSocket = new File(unixSocketPath);

		if (!unixSocket.exists()) {
			final String path = unixSocket.getAbsolutePath();
			final String warn = "Warning: Could not found unix socket path: " + path
					+ " ... Simplicity could be offline! ... Creating a unix socket file";
			System.out.println(warn);

			try {
				unixSocket.createNewFile();
			} catch (IOException e) {
				throw new IOException("Could not create the default unix socket file under the path:" + path);
			}
		}
		return unixSocket;
	}

}
