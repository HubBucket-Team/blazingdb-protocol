package com.blazingdb.protocol;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.util.Iterator;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import jnr.enxio.channels.NativeSelectorProvider;
import jnr.unixsocket.UnixServerSocket;
import jnr.unixsocket.UnixServerSocketChannel;
import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;

public class UnixService implements Runnable {
    static int MAX_BUFFER_SIZE = 4096;

    interface Actor {
        boolean rxready(IService handler);
    }

    static final class ServerActor implements Actor {
        private final UnixServerSocketChannel channel;
        private final Selector selector;

        public ServerActor(UnixServerSocketChannel channel, Selector selector) {
            this.channel = channel;
            this.selector = selector;
        }
        public final boolean rxready(IService handler) {
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

        public final boolean rxready(IService handler) {
            try {
                ByteBuffer buf = ByteBuffer.allocate(MAX_BUFFER_SIZE);
                int n = channel.read(buf);
                UnixSocketAddress remote = channel.getRemoteSocketAddress();
                if (n > 0) {
                    buf.rewind();
                    ByteBuffer response = handler.process(buf);
                    channel.write( response );
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
    
    private UnixSocketAddress address = null;
    private UnixServerSocketChannel channel = null;
    private IService handler;

    public UnixService(IService handler)  {
        this.handler = handler;
    }

    public void bind(File unixSocket) throws IOException {
        address = new UnixSocketAddress(unixSocket);
        channel = UnixServerSocketChannel.open();
    }

    @Override
    public void run() {
        try {
            Selector sel = NativeSelectorProvider.getInstance().openSelector();

            channel.configureBlocking(false);
            channel.socket().bind(address);
            channel.register(sel, SelectionKey.OP_ACCEPT, new ServerActor(channel, sel));

            while (!Thread.currentThread().isInterrupted()) {
                System.out.println("Waiting for messages");
                while (sel.select() > 0) {
                    Set<SelectionKey> keys = sel.selectedKeys();
                    Iterator<SelectionKey> iterator = keys.iterator();
                    boolean running = false;
                    boolean cancelled = false;
                    while (iterator.hasNext()) {
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
                        break;
                    }
                }
            }
            this.channel.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}