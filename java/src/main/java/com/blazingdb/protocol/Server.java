package com.blazingdb.protocol;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.util.Iterator;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.IOUtils;
 

import jnr.enxio.channels.NativeSelectorProvider;
import jnr.unixsocket.UnixServerSocket;
import jnr.unixsocket.UnixServerSocketChannel;
import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;

public class Server {
	
	static interface Actor {
        public boolean rxready(ServerHandler handler);
    }

    static final class ServerActor implements Actor {
        private final UnixServerSocketChannel channel;
        private final Selector selector;

        public ServerActor(UnixServerSocketChannel channel, Selector selector) {
            this.channel = channel;
            this.selector = selector;
        }
        public final boolean rxready(ServerHandler handler) {
            try {
                UnixSocketChannel client = channel.accept();
                client.configureBlocking(false);
                client.register(selector, SelectionKey.OP_READ, new ClientActor(client));
                return true;
            } catch (IOException ex) {
                return false;
            }
        }
    }
    static final class ClientActor implements Actor {
        private final UnixSocketChannel channel;

        public ClientActor(UnixSocketChannel channel) {
            this.channel = channel;
        }

        public final boolean rxready(ServerHandler handler) {
            try {
                ByteBuffer buf = ByteBuffer.allocate(1024);
                int n = channel.read(buf);
                UnixSocketAddress remote = channel.getRemoteSocketAddress();
                System.out.printf("Read in %d bytes from %s\n", n, remote);

                handler.accept(buf);
                
                if (n > 0) {
                    buf.flip();
                    channel.write(buf);
                    return true;
                } else if (n < 0) {
                    return false;
                }

            } catch (IOException ex) {
                ex.printStackTrace();
                return false;
            }
            return true;
        }
    }
    
    private final File unixSocket;
    UnixSocketAddress address = null;
    UnixServerSocketChannel channel = null;
    
    public Server(File unixSocket, ServerHandler handler) throws IOException {
        this.unixSocket = unixSocket;

       
        address = new UnixSocketAddress(unixSocket);
        channel = UnixServerSocketChannel.open();

        try {
            Selector sel = NativeSelectorProvider.getInstance().openSelector();
            channel.configureBlocking(false);
            channel.socket().bind(address);
            channel.register(sel, SelectionKey.OP_ACCEPT, new ServerActor(channel, sel));

            while (sel.select() > 0) {
                Set<SelectionKey> keys = sel.selectedKeys();
                Iterator<SelectionKey> iterator = keys.iterator();
                boolean running = false;
                boolean cancelled = false;
                while ( iterator.hasNext()  ) {
                    SelectionKey k = iterator.next();
                    Actor a = (Actor) k.attachment();
                    if (a.rxready(handler)) { 
                        running = true;
                    } else {
                        k.cancel();
                        cancelled = true;
                    }
                    iterator.remove();
                }
                if (!running && cancelled) {
                    System.out.println("No Actors Running any more");
                    break;
                }
            }
        } catch (IOException ex) {
            Logger.getLogger(UnixServerSocket.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println("UnixServer EXIT");
    }
 
}