import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function HealthChart({ data }) {
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{top:10,right:8,left:-18,bottom:0}}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e9eee8"/><XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill:'#758076',fontSize:12}}/><YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{fill:'#758076',fontSize:12}}/><Tooltip/><Bar dataKey="count" fill="#4e7958" radius={[5,5,0,0]} barSize={54}/></BarChart></ResponsiveContainer>
}
