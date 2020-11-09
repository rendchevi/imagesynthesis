import sys

def report_loss_simple(epoch_state, batch_state, loss_D, loss_G, duration):
    loss_D = str(loss_D.copy())[:9]
    loss_G = str(loss_G.copy())[:9]
    duration = str(duration)[:9]
    
    return sys.stdout.write('\r[Epoch] {}/{} [Batch] {}/{} [Loss D] {} [Loss G] {} [Time Elapsed] {}'.format(epoch_state[0], epoch_state[1],
                                                                                                             batch_state[0], batch_state[1],
                                                                                                             loss_D, loss_G,
                                                                                                             duration))
