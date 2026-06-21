import { useState, useEffect } from 'react';
import { ServiceStatus } from '../types';
import { orchestrator } from '../api/orchestrator';

export default function SystemStatus() {
  const [services, setServices] = useState<ServiceStatus[]>([]);

  useEffect(() => {
    setServices(orchestrator.fetchServiceStatus());

    const unsub = orchestrator.on('service_status', (data: ServiceStatus[]) => {
      setServices(data);
    });

    const interval = setInterval(() => {
      setServices(orchestrator.fetchServiceStatus());
    }, 10000);

    return () => {
      unsub();
      clearInterval(interval);
    };
  }, []);

  const getUptime = (connectedAt: string): string => {
    const diff = Date.now() - new Date(connectedAt).getTime();
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  return (
    <div className="system-status">
      {services.map((svc) => {
        const isOnline = svc.lastSeen && (Date.now() - new Date(svc.lastSeen).getTime()) < 15000;
        return (
          <div key={svc.id} className={`status-card ${isOnline ? 'online' : 'offline'}`}>
            <div className="status-card-header">
              <span className={`status-indicator ${isOnline ? 'online' : 'offline'}`} />
              <span className="status-card-name">{svc.type}</span>
            </div>
            <div className="status-card-body">
              <span className="status-card-status">{isOnline ? 'Online' : 'Offline'}</span>
              {isOnline && (
                <span className="status-card-uptime">{getUptime(svc.connectedAt)} uptime</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}